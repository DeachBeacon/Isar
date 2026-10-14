// ============================================================================
// tb_dw_conv_engine.v — randomized backpressure testbench, bit-exact compare
// Params from sim/data/params.hex (written by gen_golden_dw.py)
// ============================================================================
`timescale 1ns / 1ps

module tb_dw_conv_engine;

    integer C, H, W, STRIDE, PAD, ZP_IN, ZPOUT, MUL, SHIFT;

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

    reg [7:0] in_mem   [0:65535];
    reg [7:0] wt_mem   [0:16383];
    reg [7:0] gold_mem [0:65535];
    reg [7:0] out_mem  [0:65535];
    integer   out_n = 0;

    reg [31:0] params [0:8];

    dw_conv_engine dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .cfg_cin(C[15:0]), .cfg_h(H[15:0]), .cfg_w(W[15:0]),
        .cfg_stride(STRIDE[3:0]), .cfg_pad(PAD[3:0]),
        .cfg_mul(MUL[31:0]), .cfg_shift(SHIFT[7:0]),
        .cfg_zp_in(ZP_IN[7:0]), .cfg_zp_out(ZPOUT[7:0]),
        .wt_waddr(wt_waddr), .wt_wdata(wt_wdata), .wt_we(wt_we),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_data(out_data), .out_ready(out_ready)
    );

    always #5 clk = ~clk;

    always @(negedge clk) begin
        if (rst_n)
            out_ready <= ({$random} % 3 != 0);
    end

    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            out_mem[out_n] <= out_data;
            out_n <= out_n + 1;
        end
    end

    initial begin
        #50_000_000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end

    integer i, errors, exp_n;

    initial begin
        $readmemh("data/params.hex", params);
        C      = params[0];
        H      = params[1];
        W      = params[2];
        STRIDE = params[3];
        PAD    = params[4];
        ZP_IN  = params[5];
        ZPOUT  = params[6];
        MUL    = params[7];
        SHIFT  = params[8];
        exp_n = C * ((H + 2*PAD - 3)/STRIDE + 1) * ((W + 2*PAD - 3)/STRIDE + 1);

        $display("TB params: C=%0d H=%0d W=%0d S=%0d P=%0d ZPIN=%0d ZPOUT=%0d MUL=%0d SHIFT=%0d exp=%0d",
                 C, H, W, STRIDE, PAD, ZP_IN, ZPOUT, MUL, SHIFT, exp_n);

        $readmemh("data/input.hex", in_mem);
        $readmemh("data/weights.hex", wt_mem);
        $readmemh("data/golden.hex", gold_mem);

        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        for (i = 0; i < C * 9; i = i + 1) begin
            @(negedge clk);
            wt_waddr = i[15:0];
            wt_wdata = wt_mem[i];
            wt_we    = 1;
            @(posedge clk);
        end
        @(negedge clk);
        wt_we = 0;

        @(negedge clk);
        start = 1;
        @(posedge clk);
        @(negedge clk);
        start = 0;

        i = 0;
        while (i < C * H * W) begin
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

        @(posedge done);
        repeat (8) @(posedge clk);

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
