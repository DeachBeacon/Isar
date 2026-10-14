# build_led.tcl — 非工程模式：综合 + 实现 + 出 bitstream（headless）
# 从仓库根目录运行（build/build_led.sh 会先 cd 到仓库根）。
# 注意：仓库路径含空格，read_verilog/read_xdc 必须用相对路径，不能用绝对路径
#      （Vivado 的 read_* 对含空格的绝对路径参数会拆分成多个参数导致失败）。
set part xczu19eg-ffvc1760-2-i

read_verilog board/led_blink/led_blink.v
read_xdc    board/led_blink/led_blink.xdc

synth_design -top led_blink -part $part
opt_design
place_design
route_design

write_bitstream -force board/led_blink/led_blink.bit
report_utilization    -file board/led_blink/utilization.rpt
report_timing_summary -file board/led_blink/timing.rpt

puts "=== BITSTREAM OK ==="
