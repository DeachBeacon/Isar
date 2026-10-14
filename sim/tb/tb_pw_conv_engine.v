// ============================================================================
// tb_pw_conv_engine.v — randomized backpressure testbench, bit-exact compare
// Params come from plusargs produced by sim/scripts/gen_golden.py
// ============================================================================
`timescale 1ns / 1ps

module tb_pw_conv_engine;

    // defaults overridden by plusargs
    integer CIN   = 8;
    integer COUT  = 16;
    integer H     = 6;
    integer W     = 7;
    integer ZP_IN = 128;
    integer ZPOUT = 128;
    integer MUL   = 524288;
    integer SHIFT = 16;

    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    wire busy, done;
    reg in_valid = 0;
    reg [7:0] in_data = 0;
    wire in_ready;
    reg out_ready = 0;
    wire out_valid;
    wire [7:0] out_data;

    reg [15:0] wt_waddr = 0;
    reg [7:0]  wt_wdata = 0;
    reg        wt_we = 0;

    // stimulus / reference memories
    reg [7:0] in_mem   [0:65535];
    reg [7:0] wt_mem   [0:16383];
    reg [7:0] gold_mem [0:65535];
    reg [7:0] out_mem  [0:65535];
    integer   out_n = 0;

    pw_conv_engine dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .cfg_cin(CIN[15:0]), .cfg_cout(COUT[15:0]),
        .cfg_h(H[15:0]), .cfg_w(W[15:0]),
        .cfg_mul(MUL[31:0]), .cfg_shift(SHIFT[7:0]),
        .cfg_zp_in(ZP_IN[7:0]), .cfg_zp_out(ZPOUT[7:0]),
        .wt_waddr(wt_waddr), .wt_wdata(wt_wdata), .wt_we(wt_we),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_data(out_data), .out_ready(out_ready)
    );

    always #5 clk = ~clk;

    // random output backpressure
    always @(negedge clk) begin
        if (rst_n)
            out_ready <= ({$random} % 3 != 0);
    end

    // output collector
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            out_mem[out_n] <= out_data;
            out_n <= out_n + 1;
        end
    end

    // watchdog
    initial begin
        #50_000_000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end

    integer i, errors, exp_n;
    reg [31:0] params [0:7];

    initial begin
        // ---- params from file (CIN COUT H W ZP_IN ZP_OUT MUL SHIFT) ----
        $readmemh("data/params.hex", params);
        CIN   = params[0];
        COUT  = params[1];
        H     = params[2];
        W     = params[3];
        ZP_IN = params[4];
        ZPOUT = params[5];
        MUL   = params[6];
        SHIFT = params[7];
        exp_n = COUT * H * W;

        $display("TB params: CIN=%0d COUT=%0d H=%0d W=%0d ZPIN=%0d ZPOUT=%0d MUL=%0d SHIFT=%0d",
                 CIN, COUT, H, W, ZP_IN, ZPOUT, MUL, SHIFT);

        // ---- load files ----
        $readmemh("data/input.hex", in_mem);
        $readmemh("data/weights.hex", wt_mem);
        $readmemh("data/golden.hex", gold_mem);

        // ---- reset ----
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        // ---- load weights via debug port ----
        for (i = 0; i < CIN * COUT; i = i + 1) begin
            @(negedge clk);
            wt_waddr = i[15:0];
            wt_wdata = wt_mem[i];
            wt_we    = 1;
            @(posedge clk);
        end
        @(negedge clk);
        wt_we = 0;

        // ---- start pulse ----
        start = 1;
        @(posedge clk);
        @(negedge clk);
        start = 0;

        // ---- stream input with random gaps ----
        i = 0;
        while (i < CIN * H * W) begin
            @(negedge clk);
            if ({$random} % 4 != 0) begin
                in_valid <= 1;
                in_data  <= in_mem[i];
            end else begin
                in_valid <= 0;
            end
            @(posedge clk);
            if (in_valid && in_ready)
                i = i + 1;
        end
        @(negedge clk);
        in_valid <= 0;

        // ---- wait for completion ----
        @(posedge done);
        repeat (8) @(posedge clk);

        // ---- compare ----
        errors = 0;
        if (out_n !== exp_n) begin
            errors = errors + 1;
            $display("LENGTH MISMATCH: dut=%0d expected=%0d", out_n, exp_n);
        end
        for (i = 0; i < exp_n; i = i + 1) begin
            if (out_mem[i] !== gold_mem[i]) begin
                if (errors < 10)
                    $display("MISMATCH idx=%0d dut=%02h golden=%02h",
                             i, out_mem[i], gold_mem[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("RESULT: PASS (%0d outputs, all bit-exact)", exp_n);
        else
            $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
