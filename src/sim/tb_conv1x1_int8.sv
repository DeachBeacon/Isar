`timescale 1ns / 1ps

// DUT contract:
//   Default module name: conv1x1_int8 (override CONV1X1_DUT if necessary).
//   All ports use clk. rst_n is active low; hold it for several clock edges.
//   start is a one-cycle pulse, accepted only while idle. Latch cfg_* then.
//   One job: load Cin*Cout weights, then process npos input vectors.
//   Weight order: W[output_channel][input_channel].
//   Input order:  X[position][input_channel] (NOT ONNX NCHW memory order).
//   Output order: Y[position][output_channel].
//   INT8 inputs/weights and INT32 outputs are signed two's-complement values.
//   Y[p,o] = sum_i X[p,i]*W[o,i]. NO bias, zero-point correction, or ReLU.
//   A transfer happens only on a rising edge with valid && ready.
//   Hold y_valid, y_data and y_last during output backpressure.
//   y_last marks only the final result of the entire job.
//   busy asserts after start and remains high until all results are accepted.
//   done is a one-cycle pulse, on or after the final output handshake.
//   error must be low for every legal job in this testbench.
//   Reset aborts the job and clears busy, done, y_valid and error.
//
// Optional simulator arguments:
//   +SEED=123            Deterministic random seed (default 20261015).
//   +TIMEOUT_SCALE=32    Per-job watchdog margin (default 32).
//   +QUICK               Skip the two largest weight-matrix tests.
//
// No DUT implementation is included in this file.

`ifndef CONV1X1_DUT
`define CONV1X1_DUT conv1x1_int8
`endif

module tb_conv1x1_int8;
    localparam integer CLOCK_PERIOD_NS = 10;
    localparam integer MAX_CHANNELS = 576;
    localparam integer MAX_WEIGHT_ELEMENTS = 384 * 576;

    localparam integer PAT_DIRECTED     = 0;
    localparam integer PAT_RANDOM       = 1;
    localparam integer PAT_ZERO_WEIGHTS = 2;
    localparam integer PAT_ZERO_INPUTS  = 3;
    localparam integer PAT_MIN_VALUES   = 4;
    localparam integer PAT_EXTREMES     = 5;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [9:0] cfg_cin = '0;
    logic [9:0] cfg_cout = '0;
    logic [31:0] cfg_npos = '0;
    logic start = 1'b0;
    wire busy;
    wire done;
    wire error;

    logic signed [7:0] w_data = '0;
    logic w_valid = 1'b0;
    wire w_ready;
    logic signed [7:0] x_data = '0;
    logic x_valid = 1'b0;
    wire x_ready;
    wire signed [31:0] y_data;
    wire y_valid;
    logic y_ready = 1'b0;
    wire y_last;

    `CONV1X1_DUT dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .cfg_cin  (cfg_cin),
        .cfg_cout (cfg_cout),
        .cfg_npos (cfg_npos),
        .start    (start),
        .busy     (busy),
        .done     (done),
        .error    (error),
        .w_data   (w_data),
        .w_valid  (w_valid),
        .w_ready  (w_ready),
        .x_data   (x_data),
        .x_valid  (x_valid),
        .x_ready  (x_ready),
        .y_data   (y_data),
        .y_valid  (y_valid),
        .y_ready  (y_ready),
        .y_last   (y_last)
    );

    always #(CLOCK_PERIOD_NS / 2) clk = ~clk;

    // Separate random streams keep results independent of concurrent process order.
    logic [31:0] data_rng;
    logic [31:0] weight_gap_rng;
    logic [31:0] input_gap_rng;
    logic [31:0] ready_rng;
    integer seed_value = 20261015;
    integer timeout_scale = 32;

    logic signed [7:0] weights[];
    logic signed [7:0] inputs[];
    logic signed [31:0] expected[];

    string case_name = "idle";
    integer cin_count = 0;
    integer cout_count = 0;
    integer position_count = 0;
    integer weights_seen = 0;
    integer inputs_seen = 0;
    integer outputs_seen = 0;
    integer passed_cases = 0;
    integer aborted_cases = 0;
    longint unsigned job_cycles = 0;
    longint unsigned timeout_budget = 0;
    bit job_active = 1'b0;
    bit done_seen = 1'b0;
    bit busy_seen = 1'b0;
    bit random_backpressure = 1'b0;
    bit force_output_stall = 1'b0;
    integer stall_cycles_left = 0;

    bit was_stalled = 1'b0;
    logic signed [31:0] held_output;
    logic held_last;
    logic previous_done = 1'b0;

    function automatic logic [31:0] next_rng(input logic [31:0] state);
        logic [31:0] value;
        begin
            value = (state == 0) ? 32'h1 : state;
            value = value ^ (value << 13);
            value = value ^ (value >> 17);
            value = value ^ (value << 5);
            next_rng = value;
        end
    endfunction

    function automatic logic signed [7:0] extreme_weight(input integer index);
        begin
            case (index % 4)
                0: extreme_weight = -128;
                1: extreme_weight = 127;
                2: extreme_weight = -1;
                default: extreme_weight = 1;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] extreme_input(input integer index);
        begin
            case (index % 4)
                0: extreme_input = 127;
                1: extreme_input = -128;
                2: extreme_input = 1;
                default: extreme_input = -1;
            endcase
        end
    endfunction

    // Drive all DUT inputs on falling edges; sample handshakes on rising edges.
    always @(negedge clk) begin
        if (!rst_n) begin
            y_ready = 1'b0;
            stall_cycles_left = 0;
        end else if (force_output_stall) begin
            y_ready = 1'b0;
            stall_cycles_left = 0;
        end else if (!random_backpressure) begin
            y_ready = 1'b1;
            stall_cycles_left = 0;
        end else if (stall_cycles_left != 0) begin
            y_ready = 1'b0;
            stall_cycles_left = stall_cycles_left - 1;
        end else begin
            ready_rng = next_rng(ready_rng);
            if (ready_rng[3:0] < 4) begin
                y_ready = 1'b0;
                stall_cycles_left = 1 + (ready_rng[7:4] % 7);
            end else begin
                y_ready = 1'b1;
            end
        end
    end

    // This process remains active even when a driver is blocked waiting for ready.
    always @(posedge clk) begin
        if (!rst_n) begin
            was_stalled = 1'b0;
            previous_done = 1'b0;
        end else begin
            if ($isunknown({busy, done, error, w_ready, x_ready, y_valid}))
                $fatal(1, "[%s] X/Z on a DUT control output at %0t", case_name, $time);
            if (error !== 1'b0)
                $fatal(1, "[%s] Unexpected error for a legal configuration", case_name);
            if (done && previous_done)
                $fatal(1, "[%s] done must be a one-cycle pulse", case_name);
            previous_done = done;

            if (was_stalled) begin
                if (y_valid !== 1'b1 || y_data !== held_output || y_last !== held_last)
                    $fatal(1, "[%s] Output changed before its handshake at %0t", case_name, $time);
            end
            was_stalled = (y_valid === 1'b1 && y_ready === 1'b0);
            if (was_stalled) begin
                held_output = y_data;
                held_last = y_last;
            end

            if (job_active) begin
                job_cycles = job_cycles + 1;
                if (job_cycles > timeout_budget)
                    $fatal(1, "[%s] Timeout: cycles=%0d W=%0d/%0d X=%0d/%0d Y=%0d/%0d busy=%b",
                           case_name, job_cycles, weights_seen, weights.size(),
                           inputs_seen, inputs.size(), outputs_seen, expected.size(), busy);

                if (busy)
                    busy_seen = 1'b1;
                if (!busy_seen && job_cycles > 8)
                    $fatal(1, "[%s] busy did not assert after start", case_name);

                if (w_valid && w_ready) begin
                    weights_seen = weights_seen + 1;
                    if (weights_seen > weights.size())
                        $fatal(1, "[%s] Accepted too many weights", case_name);
                end
                if (x_valid && x_ready) begin
                    inputs_seen = inputs_seen + 1;
                    if (inputs_seen > inputs.size())
                        $fatal(1, "[%s] Accepted too many input values", case_name);
                end

                if (y_valid) begin
                    if (outputs_seen >= expected.size())
                        $fatal(1, "[%s] Unexpected extra output", case_name);
                    if (weights_seen != weights.size())
                        $fatal(1, "[%s] Output appeared before weights were loaded", case_name);
                    if (inputs_seen < ((outputs_seen / cout_count) + 1) * cin_count)
                        $fatal(1, "[%s] Output appeared before its input vector was complete", case_name);
                    if (y_last !== (outputs_seen == expected.size() - 1))
                        $fatal(1, "[%s] Incorrect y_last at output %0d", case_name, outputs_seen);
                    if (y_data !== expected[outputs_seen])
                        $fatal(1, "[%s] MISMATCH p=%0d oc=%0d: expected=%0d actual=%0d (hex=%h)",
                               case_name, outputs_seen / cout_count, outputs_seen % cout_count,
                               $signed(expected[outputs_seen]), $signed(y_data), y_data);
                    if (y_ready)
                        outputs_seen = outputs_seen + 1;
                end

                if (busy_seen && !busy && outputs_seen != expected.size())
                    $fatal(1, "[%s] busy dropped before all outputs were accepted", case_name);
                if (done) begin
                    if (done_seen)
                        $fatal(1, "[%s] Duplicate done pulse", case_name);
                    if (outputs_seen != expected.size() || inputs_seen != inputs.size() ||
                        weights_seen != weights.size())
                        $fatal(1, "[%s] done before all expected transfers completed", case_name);
                    done_seen = 1'b1;
                end
            end else begin
                if (y_valid || done || busy)
                    $fatal(1, "[%s] DUT was not idle outside a job", case_name);
            end
        end
    end

    task automatic prepare_case(
        input string name,
        input integer cin_value,
        input integer cout_value,
        input integer npos_value,
        input integer pattern
    );
        integer index;
        integer p;
        integer o;
        integer i;
        longint signed sum;
        longint signed x_value;
        longint signed w_value;
        begin
            if (cin_value < 1 || cin_value > MAX_CHANNELS ||
                cout_value < 1 || cout_value > MAX_CHANNELS || npos_value < 1 ||
                cin_value * cout_value > MAX_WEIGHT_ELEMENTS)
                $fatal(1, "Invalid testbench configuration for %s", name);

            case_name = name;
            cin_count = cin_value;
            cout_count = cout_value;
            position_count = npos_value;
            weights = new[cin_count * cout_count];
            inputs = new[cin_count * position_count];
            expected = new[cout_count * position_count];

            for (index = 0; index < weights.size(); index = index + 1) begin
                data_rng = next_rng(data_rng);
                case (pattern)
                    PAT_ZERO_WEIGHTS: weights[index] = 0;
                    PAT_MIN_VALUES:   weights[index] = -128;
                    PAT_EXTREMES:     weights[index] = extreme_weight(index);
                    default:          weights[index] = data_rng[7:0];
                endcase
            end
            for (index = 0; index < inputs.size(); index = index + 1) begin
                data_rng = next_rng(data_rng);
                case (pattern)
                    PAT_ZERO_INPUTS: inputs[index] = 0;
                    PAT_MIN_VALUES:  inputs[index] = -128;
                    PAT_EXTREMES:    inputs[index] = extreme_input(index);
                    default:         inputs[index] = data_rng[7:0];
                endcase
            end

            if (pattern == PAT_DIRECTED) begin
                if (cin_count != 3 || cout_count != 2 || position_count != 2)
                    $fatal(1, "Directed pattern requires Cin=3 Cout=2 npos=2");
                weights[0] = -2; weights[1] = 3;  weights[2] = -4;
                weights[3] = 5;  weights[4] = -6; weights[5] = 7;
                inputs[0] = -8;  inputs[1] = 9;   inputs[2] = -10;
                inputs[3] = 11;  inputs[4] = -12; inputs[5] = 13;
            end

            // Explicit 64-bit signed operands avoid an 8/16-bit reference overflow.
            for (p = 0; p < position_count; p = p + 1) begin
                for (o = 0; o < cout_count; o = o + 1) begin
                    sum = 0;
                    for (i = 0; i < cin_count; i = i + 1) begin
                        x_value = $signed(inputs[p * cin_count + i]);
                        w_value = $signed(weights[o * cin_count + i]);
                        sum = sum + x_value * w_value;
                    end
                    if (sum < -64'sd2147483648 || sum > 64'sd2147483647)
                        $fatal(1, "Reference overflow in %s", name);
                    expected[p * cout_count + o] = sum[31:0];
                end
            end

            if (pattern == PAT_DIRECTED &&
                (expected[0] !== 32'sd83 || expected[1] !== -32'sd164 ||
                 expected[2] !== -32'sd110 || expected[3] !== 32'sd218))
                $fatal(1, "Directed reference self-check failed");

            timeout_budget = 64'd2000 + timeout_scale *
                (64'd1 * weights.size() + inputs.size() + expected.size() +
                 64'd1 * cin_count * cout_count * position_count);
            $display("START %-24s Cin=%0d Cout=%0d npos=%0d", name,
                     cin_count, cout_count, position_count);
        end
    endtask

    task automatic launch_job;
        begin
            @(negedge clk);
            if (busy !== 1'b0 || done !== 1'b0 || y_valid !== 1'b0)
                $fatal(1, "[%s] DUT was not idle before start", case_name);
            weights_seen = 0;
            inputs_seen = 0;
            outputs_seen = 0;
            job_cycles = 0;
            done_seen = 1'b0;
            busy_seen = 1'b0;
            cfg_cin = cin_count;
            cfg_cout = cout_count;
            cfg_npos = position_count;
            job_active = 1'b1;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic drive_weights(input bit insert_gaps);
        integer index;
        integer gap;
        begin
            for (index = 0; index < weights.size(); index = index + 1) begin
                gap = 0;
                if (insert_gaps) begin
                    weight_gap_rng = next_rng(weight_gap_rng);
                    gap = weight_gap_rng[2:0] % 4;
                end
                repeat (gap) begin
                    @(negedge clk);
                    w_valid = 1'b0;
                end
                @(negedge clk);
                w_data = weights[index];
                w_valid = 1'b1;
                do begin
                    @(posedge clk);
                end while (w_ready !== 1'b1);
            end
            @(negedge clk);
            w_valid = 1'b0;
            w_data = '0;
        end
    endtask

    task automatic drive_inputs(input integer count, input bit insert_gaps);
        integer index;
        integer gap;
        begin
            if (count < 1 || count > inputs.size())
                $fatal(1, "Invalid input driver length");
            for (index = 0; index < count; index = index + 1) begin
                gap = 0;
                if (insert_gaps) begin
                    input_gap_rng = next_rng(input_gap_rng);
                    gap = input_gap_rng[2:0] % 4;
                end
                repeat (gap) begin
                    @(negedge clk);
                    x_valid = 1'b0;
                end
                @(negedge clk);
                x_data = inputs[index];
                x_valid = 1'b1;
                do begin
                    @(posedge clk);
                end while (x_ready !== 1'b1);
            end
            @(negedge clk);
            x_valid = 1'b0;
            x_data = '0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk);
            rst_n = 1'b0;
            job_active = 1'b0;
            start = 1'b0;
            w_valid = 1'b0;
            x_valid = 1'b0;
            w_data = '0;
            x_data = '0;
            repeat (5) @(negedge clk);
            if (busy !== 1'b0 || done !== 1'b0 || y_valid !== 1'b0 || error !== 1'b0)
                $fatal(1, "Reset did not clear DUT state");
            rst_n = 1'b1;
            repeat (3) @(negedge clk);
        end
    endtask

    task automatic run_case(
        input string name,
        input integer cin_value,
        input integer cout_value,
        input integer npos_value,
        input integer pattern,
        input bit insert_gaps,
        input bit insert_backpressure
    );
        begin
            prepare_case(name, cin_value, cout_value, npos_value, pattern);
            random_backpressure = insert_backpressure;
            force_output_stall = 1'b0;
            launch_job();
            drive_weights(insert_gaps);
            drive_inputs(inputs.size(), insert_gaps);
            while (!done_seen) @(negedge clk);
            // Keep checking after done to catch extra results and duplicate pulses.
            repeat (4) @(negedge clk);
            if (busy !== 1'b0 || done !== 1'b0 || y_valid !== 1'b0)
                $fatal(1, "[%s] DUT did not return to idle", case_name);
            job_active = 1'b0;
            passed_cases = passed_cases + 1;
            $display("PASS  %-24s outputs=%0d cycles=%0d", case_name,
                     outputs_seen, job_cycles);
            repeat (3) @(negedge clk);
        end
    endtask

    task automatic reset_with_pending_output;
        begin
            prepare_case("reset_pending_output", 24, 48, 2, PAT_RANDOM);
            random_backpressure = 1'b0;
            force_output_stall = 1'b1;
            launch_job();
            drive_weights(1'b1);
            // Feed only the first position. Its first output cannot be accepted.
            drive_inputs(cin_count, 1'b1);
            while (y_valid !== 1'b1) @(negedge clk);
            repeat (4) @(negedge clk);
            if (outputs_seen != 0 || done_seen)
                $fatal(1, "Reset scenario accidentally accepted an output");
            reset_dut();
            force_output_stall = 1'b0;
            repeat (4) @(negedge clk);
            aborted_cases = aborted_cases + 1;
            $display("PASS  reset_pending_output     pending result discarded by reset");
        end
    endtask

    initial begin
        if ($value$plusargs("SEED=%d", seed_value)) begin end
        if ($value$plusargs("TIMEOUT_SCALE=%d", timeout_scale)) begin end
        if (timeout_scale < 1)
            $fatal(1, "TIMEOUT_SCALE must be positive");
        data_rng = seed_value ^ 32'hd1342543;
        weight_gap_rng = seed_value ^ 32'h9e3779b9;
        input_gap_rng = seed_value ^ 32'h85ebca6b;
        ready_rng = seed_value ^ 32'hc2b2ae35;
        $display("conv1x1 INT8 raw-dot-product tests; seed=%0d", seed_value);

        reset_dut();
        run_case("signed_extreme",       1,   1, 1, PAT_EXTREMES,     0, 0);
        run_case("hand_calculated",      3,   2, 2, PAT_DIRECTED,     0, 0);
        run_case("zero_weights",         5,   7, 3, PAT_ZERO_WEIGHTS, 1, 1);
        run_case("zero_inputs",          7,   5, 2, PAT_ZERO_INPUTS,  1, 1);
        run_case("model_first_1x1",     24,  48, 3, PAT_RANDOM,       1, 1);
        run_case("int32_accumulation", 576,   3, 2, PAT_MIN_VALUES,   1, 1);
        run_case("model_cout_193",      96, 193, 2, PAT_RANDOM,       1, 1);
        run_case("model_cout_289",     144, 289, 1, PAT_RANDOM,       1, 1);
        run_case("model_cout_385",     192, 385, 1, PAT_RANDOM,       1, 1);

        if (!$test$plusargs("QUICK")) begin
            run_case("model_max_cout",  288, 576, 1, PAT_RANDOM,     0, 1);
            run_case("model_max_weights",576,384,1, PAT_RANDOM,     0, 1);
        end

        reset_with_pending_output();
        run_case("restart_after_reset", 3, 2, 2, PAT_DIRECTED, 1, 1);
        $display("ALL TESTS PASSED: %0d completed jobs, %0d reset-abort test",
                 passed_cases, aborted_cases);
        $finish;
    end
endmodule
