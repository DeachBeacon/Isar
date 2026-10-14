# led_blink.xdc — 引脚约束（从 Z19 用户手册 6.4 节 + 出厂工程提取）
# 器件：xczu19eg-ffvc1760-2-i（ALINX Z19 / AXU19EG）

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# PL 差分 200MHz 时钟 PL_CLK0（BANK64，全局时钟 MRCC）
set_property IOSTANDARD DIFF_SSTL12 [get_ports sys_clk_p]
set_property IOSTANDARD DIFF_SSTL12 [get_ports sys_clk_n]
set_property PACKAGE_PIN H24 [get_ports sys_clk_p]
set_property PACKAGE_PIN G25 [get_ports sys_clk_n]
create_clock -period 5.000 -name sys_clk -waveform {0.000 2.500} [get_ports sys_clk_p]

# 板载 PL LED
set_property PACKAGE_PIN B3 [get_ports {pl_led_tri_o}]
set_property IOSTANDARD LVCMOS33 [get_ports {pl_led_tri_o}]
