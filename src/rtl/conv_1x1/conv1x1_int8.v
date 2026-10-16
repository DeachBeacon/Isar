`timescale 1ns / 1ps
//=============================================================================
// conv1x1_int8.v -- 1x1 (point-wise) INT8 convolution dot-product core
//
//   Y[p][o] = SUM(i=0..Cin-1) X[p][i] * W[o][i]
//
//   * INT8 signed features and weights, INT32 signed raw dot product output.
//   * One job: load the full Cin*Cout weight matrix once, then process
//     cfg_npos input vectors and emit Cout results per vector.
//   * Output is the raw dot product S only. Zero-point correction, bias,
//     scale and requantization are applied by software afterwards.
//
// Data order (linear index):
//   weights : o*Cin + i    (output channel first)
//   inputs  : p*Cin + i    (position first)
//   outputs : p*Cout + o   (position first)
//
// Handshake: a transfer completes on a rising clock edge with valid && ready.
// While an output is back-pressured (y_valid=1, y_ready=0) y_valid, y_data and
// y_last are held stable. done is a one-cycle pulse after the final output of
// the job has been accepted; reset aborts the job and clears all state.
//
// Datapath: one signed INT8 x INT8 multiplier (int8_mult.v), one signed INT32
// accumulator, one weight RAM (MAX_WEIGHTS x 8) and a per-position input RAM
// (MAX_CIN x 8). The weight matrix is written once per job and reused for every
// output channel and every position; the input vector of one position is
// cached and reused for all output channels.
//
// MAC timing: the RAM read of X[i]/W[o*Cin+i] is registered, so data for index
// i appears one cycle after the address is issued. The sweep counter runs
// 0..Cin: during cycle i (i>=1) the product of index i-1 is accumulated, and
// the cycle with i==Cin accumulates the final product before the result is
// presented. A channel therefore costs Cin+1 cycles.
//
// NOTE: compile together with rtl/int8_mult.v.
//=============================================================================

module conv1x1_int8 #(
    parameter integer MAX_WEIGHTS = 221184, // 384*576 = 216 KiB (largest model 1x1 matrix)
    parameter integer MAX_CIN     = 576,
    parameter integer MAX_COUT    = 576
)(
    input  wire               clk,
    input  wire               rst_n,          // synchronous, active low
    input  wire [9:0]         cfg_cin,        // real input channels, e.g. 24
    input  wire [9:0]         cfg_cout,       // real output channels, e.g. 48
    input  wire [31:0]        cfg_npos,       // positions to process this job
    input  wire               start,          // one-cycle pulse, idle only
    output wire               busy,
    output wire               done,
    output wire               error,          // illegal config, sticky until reset / valid start
    input  wire signed [7:0]  w_data,
    input  wire               w_valid,
    output wire               w_ready,
    input  wire signed [7:0]  x_data,
    input  wire               x_valid,
    output wire               x_ready,
    output wire signed [31:0] y_data,
    output wire               y_valid,
    input  wire               y_ready,
    output wire               y_last
);

    localparam integer WADDR_W = $clog2(MAX_WEIGHTS); // 18 for 221184
    localparam integer XADDR_W = $clog2(MAX_CIN);     // 10 for 576

    localparam [2:0] ST_IDLE    = 3'd0,
                     ST_WEIGHTS = 3'd1,
                     ST_INPUTS  = 3'd2,
                     ST_MAC     = 3'd3,
                     ST_OUT     = 3'd4,
                     ST_DONE    = 3'd5;

    reg [2:0] state;

    // Latched configuration (sampled when start is accepted).
    reg [9:0]  cin_r;
    reg [9:0]  cout_r;
    reg [31:0] npos_r;
    reg [19:0] wtotal_m1;   // Cin*Cout - 1

    // Counters / pointers.
    reg [19:0] wcount;      // weights accepted so far
    reg [9:0]  xcount;      // inputs accepted for the current position
    reg [9:0]  oc;          // output channel being computed
    reg [9:0]  i;           // MAC sweep index, runs 0..Cin
    reg [31:0] pos;         // position being processed
    reg [19:0] base;        // weight base address of the current output channel

    // Datapath.
    reg  signed [31:0] acc;
    reg  signed [7:0]  xq, wq;  // registered RAM read data
    wire signed [15:0] prod;

    reg error_r;

    //--------------------------------------------------------------------
    // Configuration check (evaluated when start is accepted)
    //--------------------------------------------------------------------
    wire [19:0] cfg_wtotal = {10'd0, cfg_cin} * {10'd0, cfg_cout};
    wire cfg_bad = (cfg_cin  == 10'd0)   ||
                   (cfg_cout == 10'd0)   ||
                   (cfg_npos == 32'd0)   ||
                   (cfg_cin  > MAX_CIN)  ||
                   (cfg_cout > MAX_COUT) ||
                   (cfg_wtotal > MAX_WEIGHTS);

    //--------------------------------------------------------------------
    // Single signed 8x8 multiplier (shared by all channels/positions)
    //--------------------------------------------------------------------
    wire mac_en = (state == ST_MAC) && (i != 10'd0);

    int8_mult u_mult (
        .en (mac_en),
        .a  (xq),
        .b  (wq),
        .p  (prod)
    );

    //--------------------------------------------------------------------
    // Memories: no reset on contents (BRAM friendly), fully written before use
    //--------------------------------------------------------------------
    reg signed [7:0] w_ram [0:MAX_WEIGHTS-1];
    reg signed [7:0] x_ram [0:MAX_CIN-1];

    wire        w_we    = (state == ST_WEIGHTS) && w_valid && w_ready;
    wire [19:0] w_waddr = wcount;
    wire        w_re    = (state == ST_MAC) && (i < cin_r);
    wire [19:0] w_raddr = base + {10'd0, i};

    wire        x_we    = (state == ST_INPUTS) && x_valid && x_ready;
    wire [9:0]  x_waddr = xcount;
    wire        x_re    = (state == ST_MAC) && (i < cin_r);
    wire [9:0]  x_raddr = i;

    always @(posedge clk) begin
        if (w_we) w_ram[w_waddr[WADDR_W-1:0]] <= w_data;
    end

    always @(posedge clk) begin
        if (w_re) wq <= w_ram[w_raddr[WADDR_W-1:0]];
    end

    always @(posedge clk) begin
        if (x_we) x_ram[x_waddr[XADDR_W-1:0]] <= x_data;
    end

    always @(posedge clk) begin
        if (x_re) xq <= x_ram[x_raddr[XADDR_W-1:0]];
    end

    //--------------------------------------------------------------------
    // Main control FSM
    //--------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            cin_r     <= 10'd0;
            cout_r    <= 10'd0;
            npos_r    <= 32'd0;
            wtotal_m1 <= 20'd0;
            wcount    <= 20'd0;
            xcount    <= 10'd0;
            oc        <= 10'd0;
            i         <= 10'd0;
            pos       <= 32'd0;
            base      <= 20'd0;
            acc       <= 32'sd0;
            error_r   <= 1'b0;
        end else begin
            case (state)
                //----------------------------------------------------
                // Idle: accept start, latch configuration
                ST_IDLE: begin
                    if (start) begin
                        cin_r  <= cfg_cin;
                        cout_r <= cfg_cout;
                        npos_r <= cfg_npos;
                        if (cfg_bad) begin
                            error_r <= 1'b1;   // illegal config: flag, stay idle
                        end else begin
                            error_r   <= 1'b0;
                            wtotal_m1 <= cfg_wtotal - 20'd1;
                            wcount    <= 20'd0;
                            xcount    <= 10'd0;
                            oc        <= 10'd0;
                            i         <= 10'd0;
                            pos       <= 32'd0;
                            base      <= 20'd0;
                            acc       <= 32'sd0;
                            state     <= ST_WEIGHTS;
                        end
                    end
                end

                //----------------------------------------------------
                // Accept Cin*Cout weights (order: o*Cin + i)
                ST_WEIGHTS: begin
                    if (w_valid && w_ready) begin
                        if (wcount == wtotal_m1)
                            state <= ST_INPUTS;
                        else
                            wcount <= wcount + 20'd1;
                    end
                end

                //----------------------------------------------------
                // Accept the Cin inputs of one position, then start MAC
                ST_INPUTS: begin
                    if (x_valid && x_ready) begin
                        if (xcount == (cin_r - 10'd1)) begin
                            xcount <= 10'd0;
                            oc     <= 10'd0;
                            base   <= 20'd0;
                            i      <= 10'd0;
                            acc    <= 32'sd0;
                            state  <= ST_MAC;
                        end else begin
                            xcount <= xcount + 10'd1;
                        end
                    end
                end

                //----------------------------------------------------
                // Cin+1 cycles: accumulate X[i]*W[base+i] for i = 0..Cin.
                // During cycle i, the registered read data corresponds to
                // index i-1, so the last product is included on the cycle
                // where i == Cin, right before the result is presented.
                ST_MAC: begin
                    if (mac_en)
                        acc <= acc + prod;

                    if (i < cin_r)
                        i <= i + 10'd1;

                    if (i == cin_r)
                        state <= ST_OUT;
                end

                //----------------------------------------------------
                // Present one INT32 result, hold it until accepted
                ST_OUT: begin
                    if (y_ready) begin
                        if (oc == (cout_r - 10'd1)) begin
                            if (pos == (npos_r - 32'd1)) begin
                                state <= ST_DONE;
                            end else begin
                                pos   <= pos + 32'd1;
                                state <= ST_INPUTS;
                            end
                        end else begin
                            // Next output channel reuses the cached input vector
                            oc    <= oc + 10'd1;
                            base  <= base + {10'd0, cin_r};
                            i     <= 10'd0;
                            acc   <= 32'sd0;
                            state <= ST_MAC;
                        end
                    end
                end

                //----------------------------------------------------
                ST_DONE: begin
                    state <= ST_IDLE;          // one-cycle done pulse
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------
    // Outputs
    //--------------------------------------------------------------------
    assign busy    = (state != ST_IDLE);
    assign done    = (state == ST_DONE);
    assign error   = error_r;
    assign w_ready = (state == ST_WEIGHTS);
    assign x_ready = (state == ST_INPUTS);
    assign y_valid = (state == ST_OUT);
    assign y_data  = acc;
    assign y_last  = (state == ST_OUT) &&
                     (oc == (cout_r - 10'd1)) &&
                     (pos == (npos_r - 32'd1));

endmodule
