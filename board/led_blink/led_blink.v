// ============================================================================
// led_blink.v — 最小上板冒烟测试（纯 PL，无需 PS）
// 功能：差分 200MHz PL 时钟（PL_CLK0_P/N）经 IBUFDS 后分频，驱动板载 LED 闪烁。
// 用途：验证 ① 开发板供电与 PL 时钟正常 ② Vivado 工具链能出 bitstream
//       ③ JTAG 下载链路正常。这是 W1 平台闭环的第一步。
// 时钟：PL_CLK0_P=H24, PL_CLK0_N=G25（DIFF_SSTL12，BANK64）
// ============================================================================
`timescale 1ns / 1ps

module led_blink (
    input  wire sys_clk_p,     // H24
    input  wire sys_clk_n,     // G25
    output wire pl_led_tri_o    // B3
);

    wire clk;
    IBUFDS #(
        .DIFF_TERM  ("FALSE"),
        .IOSTANDARD ("DIFF_SSTL12")
    ) u_ibufds (
        .I  (sys_clk_p),
        .IB (sys_clk_n),
        .O  (clk)
    );

    reg [26:0] cnt;
    always @(posedge clk) begin
        cnt <= cnt + 27'd1;
    end

    // 2^27 / 200MHz = 0.671s 半周期
    assign pl_led_tri_o = cnt[26];

endmodule
