// ============================================================================
// pw_conv_engine.v — 1x1 Pointwise Convolution Engine (INT8, quantized)
//
// Project : Isar — MobileViTv2 INT8 accelerator on AXU19EG (Zynq US+)
// Stage   : W2 baseline (correctness-first; streaming 256-MAC version is the
//           later optimization and will be reported in the compare table)
//
// Function:
//   out[oc][oy][ox] = clamp( requant( sum_ci (in[ci][oy][ox] - zp_in)
//                                      * w[oc][ci] ) + zp_out )
//   requant(acc)    = (acc * M + round) >>> shift,  round = 1 << (shift-1)
//
// Memory layouts (byte addresses):
//   input  stream : CHW order, addr = ci*H*W + oy*W + ox        (uint8)
//   weight BRAM   : addr = oc*CIN + ci                           (int8, 2's complement)
//   output stream : CHW order, addr = oc*H*W + oy*W + ox         (uint8)
//
// Interfaces:
//   in_*  : AXI-Stream-style valid/ready handshake, 8-bit payload
//   out_* : AXI-Stream-style valid/ready handshake, 8-bit payload
//   cfg_* : layer descriptor, must be stable while start pulses
//   wt_*  : weight BRAM write port (setup phase, before start)
//
// Timing (this baseline):
//   load  : 1 byte/cycle; compute : 2 cycles per MAC; emit : 1/accepted cycle
// ============================================================================
`timescale 1ns / 1ps

module pw_conv_engine #(
    parameter IN_DEPTH = 65536,          // input frame BRAM depth (bytes)
    parameter WT_DEPTH = 16384           // weight BRAM depth (bytes)
)(
    input  wire               clk,
    input  wire               rst_n,

    // ---- control / status ----
    input  wire               start,      // 1-cycle pulse
    output reg                busy,
    output reg                done,       // 1-cycle pulse

    // ---- layer descriptor (written before start) ----
    input  wire [15:0]        cfg_cin,
    input  wire [15:0]        cfg_cout,
    input  wire [15:0]        cfg_h,
    input  wire [15:0]        cfg_w,
    input  wire signed [31:0] cfg_mul,    // quant multiplier M  (>=1, <= 2^31)
    input  wire [7:0]         cfg_shift,  // right shift          (>=1)
    input  wire [7:0]         cfg_zp_in,  // input zero point
    input  wire [7:0]         cfg_zp_out, // output zero point

    // ---- weight BRAM write port (setup) ----
    input  wire [15:0]        wt_waddr,
    input  wire [7:0]         wt_wdata,
    input  wire               wt_we,

    // ---- input stream ----
    input  wire               in_valid,
    input  wire [7:0]         in_data,
    output reg                in_ready,

    // ---- output stream ----
    output reg                out_valid,
    output reg  [7:0]         out_data,
    input  wire               out_ready
);

    // ---------------- FSM ----------------
    localparam S_IDLE = 3'd0;
    localparam S_LOAD = 3'd1;
    localparam S_READ = 3'd2;   // present BRAM read addresses
    localparam S_MAC  = 3'd3;   // x_d/w_d valid, accumulate
    localparam S_OUT  = 3'd4;   // hold output until accepted

    reg [2:0] state;

    // ---------------- memories ----------------
    reg [7:0] bram_in [0:IN_DEPTH-1];
    reg [7:0] bram_wt [0:WT_DEPTH-1];

    // weight BRAM write port (own block: avoids same-block array read/write)
    reg  [15:0] rd_xaddr, rd_waddr;
    reg  [7:0]  x_d, w_d;

    always @(posedge clk) begin
        if (wt_we)
            bram_wt[wt_waddr] <= wt_wdata;
    end

    // synchronous read port (registered outputs)
    always @(posedge clk) begin
        x_d <= bram_in[rd_xaddr];
        w_d <= bram_wt[rd_waddr];
    end

    // ---------------- datapath registers ----------------
    reg [31:0] in_cnt;                   // bytes loaded so far
    reg [31:0] in_total;                 // CIN*H*W
    reg [31:0] hw;                       // H*W
    reg [15:0] oc, oy, ox, ci;
    reg [31:0] xaddr;                    // current input byte addr
    reg [31:0] pix_base;                 // addr of pixel (oy,ox) in channel 0
    reg [31:0] waddr_base;               // oc*CIN
    reg [31:0] waddr;                    // current weight addr
    reg signed [31:0] acc;

    // 32-bit zero extension helpers (Verilog keeps product width at operand
    // width, so widen operands before multiplying)
    wire [31:0] cin32 = {16'd0, cfg_cin};
    wire [31:0] h32   = {16'd0, cfg_h};
    wire [31:0] w32   = {16'd0, cfg_w};
    wire [31:0] hw32  = h32 * w32;
    wire [31:0] cin_x_hw = cin32 * hw32;

    // ---------------- MAC / requantize combinational ----------------
    wire signed [8:0]  x_eff = $signed({1'b0, x_d}) - $signed({1'b0, cfg_zp_in});
    wire signed [16:0] term  = x_eff * $signed(w_d);
    wire signed [31:0] acc_next = acc + term;

    wire signed [63:0] prod    = acc_next * cfg_mul;
    // NOTE: shift-amount expression must NOT be truncated. Writing
    //   (cfg_shift - 1'd0)  collapses the subtraction to 1 bit and yields
    //   the wrong rounding constant. Use an explicit 32-bit constant.
    wire signed [63:0] rounded = prod + (64'sd1 <<< (cfg_shift - 8'd1));
    wire signed [63:0] shifted = rounded >>> cfg_shift;
    wire signed [63:0] val     = shifted + $signed({{56{1'b0}}, cfg_zp_out});

    reg [7:0] out_q;
    always @* begin
        if      (val < 64'sd0)   out_q = 8'd0;
        else if (val > 64'sd255) out_q = 8'd255;
        else                     out_q = val[7:0];
    end

    // ---------------- next-pixel index logic ----------------
    wire last_col = (ox == cfg_w - 16'd1);
    wire last_row = (oy == cfg_h - 16'd1);
    wire last_och = (oc == cfg_cout - 16'd1);
    wire last_elem = last_col && last_row && last_och;

    reg [15:0] nx_ox, nx_oy, nx_oc;
    reg [31:0] nx_pix, nx_wbase;
    always @* begin
        nx_ox   = ox + 16'd1;
        nx_oy   = oy;
        nx_oc   = oc;
        nx_pix  = pix_base + 32'd1;      // (oy,ox+1) or start of next row
        nx_wbase= waddr_base;
        if (last_col) begin
            nx_ox = 16'd0;
            if (last_row) begin
                nx_oy    = 16'd0;
                nx_oc    = oc + 16'd1;
                nx_pix   = 32'd0;
                nx_wbase = waddr_base + {16'd0, cfg_cin};
            end else begin
                nx_oy = oy + 16'd1;
            end
        end
    end

    // ---------------- main FSM ----------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            in_ready   <= 1'b0;
            out_valid  <= 1'b0;
            out_data   <= 8'd0;
            rd_xaddr   <= 16'd0;
            rd_waddr   <= 16'd0;
            oc <= 16'd0; oy <= 16'd0; ox <= 16'd0; ci <= 16'd0;
            xaddr <= 32'd0; pix_base <= 32'd0;
            waddr_base <= 32'd0; waddr <= 32'd0;
            acc <= 32'sd0; in_cnt <= 32'd0;
            hw <= 32'd0; in_total <= 32'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy     <= 1'b1;
                        in_cnt   <= 32'd0;
                        hw       <= hw32;
                        in_total <= cin_x_hw;
                        oc <= 16'd0; oy <= 16'd0; ox <= 16'd0; ci <= 16'd0;
                        acc <= 32'sd0;
                        pix_base <= 32'd0; xaddr <= 32'd0;
                        waddr_base <= 32'd0; waddr <= 32'd0;
                        rd_xaddr <= 16'd0; rd_waddr <= 16'd0;
                        in_ready <= 1'b1;
                        state    <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (in_valid && in_ready) begin
                        bram_in[in_cnt] <= in_data;
                        in_cnt <= in_cnt + 32'd1;
                        if (in_cnt == in_total - 32'd1) begin
                            in_ready <= 1'b0;
                            state    <= S_READ;
                        end
                    end
                end

                S_READ: begin
                    // addresses already presented; data captured at this edge
                    state <= S_MAC;
                end

                S_MAC: begin
                    if (ci == cfg_cin - 16'd1) begin
                        // last channel: requantize and emit
                        out_data  <= out_q;
                        out_valid <= 1'b1;
                        acc       <= 32'sd0;
                        state     <= S_OUT;
                    end else begin
                        acc       <= acc_next;
                        ci        <= ci + 16'd1;
                        xaddr     <= xaddr + hw;
                        rd_xaddr  <= rd_xaddr + hw[15:0];
                        waddr     <= waddr + 32'd1;
                        rd_waddr  <= rd_waddr + 16'd1;
                        state     <= S_READ;
                    end
                end

                S_OUT: begin
                    if (out_ready) begin
                        out_valid <= 1'b0;
                        if (last_elem) begin
                            done  <= 1'b1;
                            busy  <= 1'b0;
                            state <= S_IDLE;
                        end else begin
                            ox        <= nx_ox;
                            oy        <= nx_oy;
                            oc        <= nx_oc;
                            pix_base  <= nx_pix;
                            waddr_base<= nx_wbase;
                            ci        <= 16'd0;
                            xaddr     <= nx_pix;
                            waddr     <= nx_wbase;
                            rd_xaddr  <= nx_pix[15:0];
                            rd_waddr  <= nx_wbase[15:0];
                            state     <= S_READ;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
