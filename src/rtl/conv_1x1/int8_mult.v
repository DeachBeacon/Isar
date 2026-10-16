`timescale 1ns / 1ps
module int8_mult #(
    parameter WIDTH = 8
)(
    input  wire en,
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    output wire signed [2*WIDTH-1:0] p
);
    assign p = en ? a * b : $signed({2*WIDTH{1'b0}});
    
endmodule //int8_mult