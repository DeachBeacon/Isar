module Multiplier # (
    parameter WIDTH = 8   // 默认值8bit，例化可重写
)(
    input wire clk,
    input wire rst_n,
    input wire en,
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    output wire [WIDTH*2-1:0]   result
);

// // 模块内部可以继续用 localparam（不能外部修改）
// localparam MAX_VAL = (1 << WIDTH) - 1;

assign result = (en) ? a * b : {WIDTH*2{1'b0}};

endmodule //Multiplier
