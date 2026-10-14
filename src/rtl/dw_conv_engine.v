// ============================================================================
// dw_conv_engine.v — 3x3 Depthwise Convolution Engine (INT8, quantized)
//
// Project : Isar — MobileViTv2 INT8 accelerator on AXU19EG (Zynq US+)
// Stage   : W2 baseline (correctness-first; streaming line-buffer version is
//           the later optimization, reported in the compare table)
//
// Function (per output channel oc, same channel in == out for DW):
//   out[oc][oy][ox] = clamp( requant( sum_ky,kx in[oc][iy][ix]*w[oc][ky][kx] )
//                             + zp_out )
//   with iy = oy*stride - pad + ky,  ix = ox*stride - pad + kx
//   out-of-range taps are zero-padded (contribute 0).
//
// Memory layouts (byte addresses):
//   input  stream : CHW, addr = ci*H*W + y*W + x          (uint8)
//   weight BRAM   : addr = oc*9 + ky*3 + kx               (int8, 2's comp)
//   output stream : CHW, addr = oc*Hout*Wout + oy*Wout + ox (uint8)
//
// Timing (baseline): load 1 B/cycle; compute 2 cycles/tap, 9 taps/pixel.
// ============================================================================
`timescale 1ns / 1ps

module dw_conv_engine #(
    parameter IN_DEPTH = 65536,
    parameter WT_DEPTH = 16384
)(
    input  wire               clk,
    input  wire               rst_n,

    input  wire               start,
    output reg                busy,
    output reg                done,

    input  wire [15:0]        cfg_cin,     // = C = cout for DW
    input  wire [15:0]        cfg_h,
    input  wire [15:0]        cfg_w,
    input  wire [3:0]         cfg_stride,  // 1 or 2
    input  wire [3:0]         cfg_pad,     // 0 or 1
    input  wire signed [31:0] cfg_mul,
    input  wire [7:0]         cfg_shift,
    input  wire [7:0]         cfg_zp_in,
    input  wire [7:0]         cfg_zp_out,

    input  wire [15:0]        wt_waddr,
    input  wire [7:0]         wt_wdata,
    input  wire               wt_we,

    input  wire               in_valid,
    input  wire [7:0]         in_data,
    output reg                in_ready,

    output reg                out_valid,
    output reg  [7:0]         out_data,
    input  wire               out_ready
);

    // ---------------- FSM ----------------
    localparam S_IDLE = 3'd0;
    localparam S_LOAD = 3'd1;
    localparam S_READ = 3'd2;
    localparam S_MAC  = 3'd3;
    localparam S_OUT  = 3'd4;

    reg [2:0] state;

    // ---------------- memories ----------------
    reg [7:0] bram_in [0:IN_DEPTH-1];
    reg [7:0] bram_wt [0:WT_DEPTH-1];

    reg [15:0] rd_xaddr, rd_waddr;
    reg [7:0]  x_d, w_d;
    reg        tap_valid_r;

    // write port: own block (see skill/pitfalls.md #2)
    always @(posedge clk) begin
        if (wt_we)
            bram_wt[wt_waddr] <= wt_wdata;
    end

    // read port: own block
    always @(posedge clk) begin
        x_d <= bram_in[rd_xaddr];
        w_d <= bram_wt[rd_waddr];
    end

    // ---------------- datapath registers ----------------
    reg [31:0] in_cnt;
    reg [31:0] in_total;   // C*H*W
    reg [31:0] hw;         // H*W
    reg [15:0] hout, wout; // output dims
    reg [15:0] oc, oy, ox;
    reg [1:0]  ky, kx;
    reg signed [31:0] acc;
    reg [3:0]  stride, pad;

    wire [31:0] cin32 = {16'd0, cfg_cin};
    wire [31:0] h32   = {16'd0, cfg_h};
    wire [31:0] w32   = {16'd0, cfg_w};
    wire [31:0] hw32  = h32 * w32;
    wire [31:0] total32 = cin32 * hw32;

    // ---------------- tap address computation (combinational) ----------------
    wire [31:0] oy32 = {16'd0, oy};
    wire [31:0] ox32 = {16'd0, ox};
    wire [31:0] oc32 = {16'd0, oc};
    wire [31:0] ky32 = {30'd0, ky};
    wire [31:0] kx32 = {30'd0, kx};
    wire [31:0] st32 = {28'd0, stride};
    wire [31:0] pd32 = {28'd0, pad};

    wire signed [31:0] iy = $signed(oy32 * st32 - pd32 + ky32);
    wire signed [31:0] ix = $signed(ox32 * st32 - pd32 + kx32);
    wire tap_valid = (iy >= 32'sd0) && (iy < $signed(h32)) &&
                     (ix >= 32'sd0) && (ix < $signed(w32));
    wire [31:0] in_addr = oc32 * hw32 + iy * w32 + ix;   // valid only if tap_valid
    wire [31:0] w_addr  = oc32 * 32'd9 + ky32 * 32'd3 + kx32;

    // clamp address so out-of-range reads never hit bad BRAM indices
    wire [31:0] in_addr_safe = tap_valid ? in_addr : 32'd0;

    // ---------------- MAC / requantize (same formula as pw engine) ----------------
    wire signed [8:0]  x_eff = $signed({1'b0, x_d}) - $signed({1'b0, cfg_zp_in});
    wire signed [16:0] term  = x_eff * $signed(w_d);
    wire signed [31:0] acc_add = tap_valid_r ? (acc + term) : acc;

    wire signed [63:0] prod    = acc_add * cfg_mul;
    wire signed [63:0] rounded = prod + (64'sd1 <<< (cfg_shift - 8'd1));
    wire signed [63:0] shifted = rounded >>> cfg_shift;
    wire signed [63:0] val     = shifted + $signed({{56{1'b0}}, cfg_zp_out});

    reg [7:0] out_q;
    always @* begin
        if      (val < 64'sd0)   out_q = 8'd0;
        else if (val > 64'sd255) out_q = 8'd255;
        else                     out_q = val[7:0];
    end

    // ---------------- next-tap / next-pixel logic ----------------
    wire last_tap = (ky == 2'd2) && (kx == 2'd2);
    wire last_col = (ox == wout - 16'd1);
    wire last_row = (oy == hout - 16'd1);
    wire last_och = (oc == cfg_cin - 16'd1);
    wire last_elem = last_col && last_row && last_och;

    reg [1:0]  nx_ky, nx_kx;
    reg [15:0] nx_ox, nx_oy, nx_oc;
    always @* begin
        nx_ky = ky; nx_kx = kx;
        nx_ox = ox; nx_oy = oy; nx_oc = oc;
        if (kx == 2'd2) begin
            nx_kx = 2'd0;
            if (ky == 2'd2) begin
                nx_ky = 2'd0;                 // tap wrap -> next pixel
                if (last_col) begin
                    nx_ox = 16'd0;
                    if (last_row) begin
                        nx_oy = 16'd0;
                        nx_oc = oc + 16'd1;
                    end else begin
                        nx_oy = oy + 16'd1;
                    end
                end else begin
                    nx_ox = ox + 16'd1;
                end
            end else begin
                nx_ky = ky + 2'd1;
            end
        end else begin
            nx_kx = kx + 2'd1;
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
            tap_valid_r<= 1'b0;
            oc <= 16'd0; oy <= 16'd0; ox <= 16'd0;
            ky <= 2'd0;  kx <= 2'd0;
            acc <= 32'sd0; in_cnt <= 32'd0;
            hw <= 32'd0; in_total <= 32'd0;
            hout <= 16'd0; wout <= 16'd0;
            stride <= 4'd0; pad <= 4'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy     <= 1'b1;
                        in_cnt   <= 32'd0;
                        hw       <= hw32;
                        in_total <= total32;
                        stride   <= cfg_stride;
                        pad      <= cfg_pad;
                        hout     <= (cfg_h + 2*cfg_pad - 16'd3) / {16'd0,cfg_stride} + 16'd1;
                        wout     <= (cfg_w + 2*cfg_pad - 16'd3) / {16'd0,cfg_stride} + 16'd1;
                        oc <= 16'd0; oy <= 16'd0; ox <= 16'd0;
                        ky <= 2'd0;  kx <= 2'd0;
                        acc <= 32'sd0;
                        in_ready <= 1'b1;
                        state    <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (in_valid && in_ready) begin
                        bram_in[in_cnt] <= in_data;
                        in_cnt <= in_cnt + 32'd1;
                        if (in_cnt == in_total - 32'd1) begin
                            in_ready    <= 1'b0;
                            rd_xaddr    <= in_addr_safe[15:0];
                            rd_waddr    <= w_addr[15:0];
                            tap_valid_r <= tap_valid;
                            state       <= S_READ;
                        end
                    end
                end

                S_READ: begin
                    state <= S_MAC;
                end

                S_MAC: begin
                    acc <= acc_add;
                    if (last_tap) begin
                        // requantize + emit
                        out_data  <= out_q;
                        out_valid <= 1'b1;
                        acc       <= 32'sd0;
                        state     <= S_OUT;
                    end else begin
                        ky        <= nx_ky;
                        kx        <= nx_kx;
                        // latch NEXT tap address (comb uses updated ky/kx next cycle)
                        // NB: nx address is computed from nx_ky/nx_kx below
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
                            ox <= nx_ox;
                            oy <= nx_oy;
                            oc <= nx_oc;
                            ky <= 2'd0;
                            kx <= 2'd0;
                            acc <= 32'sd0;
                            state <= S_READ;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // latch NEXT tap address: recompute from nx indices (registered comb path)
    wire [31:0] nx_oy32 = {16'd0, nx_oy};
    wire [31:0] nx_ox32 = {16'd0, nx_ox};
    wire [31:0] nx_oc32 = {16'd0, nx_oc};
    wire [31:0] nx_ky32 = {30'd0, nx_ky};
    wire [31:0] nx_kx32 = {30'd0, nx_kx};
    wire signed [31:0] nx_iy = $signed(nx_oy32 * st32 - pd32 + nx_ky32);
    wire signed [31:0] nx_ix = $signed(nx_ox32 * st32 - pd32 + nx_kx32);
    wire nx_valid = (nx_iy >= 32'sd0) && (nx_iy < $signed(h32)) &&
                    (nx_ix >= 32'sd0) && (nx_ix < $signed(w32));
    wire [31:0] nx_in_addr = nx_oc32 * hw32 + nx_iy * w32 + nx_ix;
    wire [31:0] nx_w_addr  = nx_oc32 * 32'd9 + nx_ky32 * 32'd3 + nx_kx32;
    wire [31:0] nx_in_addr_safe = nx_valid ? nx_in_addr : 32'd0;

    // next-PIXEL tap0 address (ky=kx=0 at new pixel)
    wire signed [31:0] np_iy = $signed(nx_oy32 * st32 - pd32 + 32'd0);
    wire signed [31:0] np_ix = $signed(nx_ox32 * st32 - pd32 + 32'd0);
    wire np_valid = (np_iy >= 32'sd0) && (np_iy < $signed(h32)) &&
                    (np_ix >= 32'sd0) && (np_ix < $signed(w32));
    wire [31:0] np_in_addr = nx_oc32 * hw32 + np_iy * w32 + np_ix;
    wire [31:0] np_w_addr  = nx_oc32 * 32'd9;
    wire [31:0] np_in_addr_safe = np_valid ? np_in_addr : 32'd0;

    always @(posedge clk) begin
        if (state == S_MAC && !last_tap) begin
            rd_xaddr    <= nx_in_addr_safe[15:0];
            rd_waddr    <= nx_w_addr[15:0];
            tap_valid_r <= nx_valid;
        end else if (state == S_OUT && out_ready && !last_elem) begin
            // new pixel, tap0 of (nx_oc, nx_oy, nx_ox)
            rd_xaddr    <= np_in_addr_safe[15:0];
            rd_waddr    <= np_w_addr[15:0];
            tap_valid_r <= np_valid;
        end
    end

endmodule
