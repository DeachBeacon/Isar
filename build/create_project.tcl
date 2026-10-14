# create_project.tcl — 在 Vivado 2025.2 里创建本项目的 RTL 工程
#
# 用途：给上板队友一个"一键建工程"的起点。创建后：
#   - 器件已设好 xczu19eg-ffvc1760-2-i
#   - 导入了自研 RTL（pw/dw 引擎）、出厂约束、出厂 ip_repo
#   - 之后在 GUI 里新建 Block Design 加 CIPS + DDR4（见 docs/platform_bringup.md）
#
# 用法（在 Vivado Tcl Console 或命令行）：
#   cd <仓库根>
#   vivado -mode tcl -source build/create_project.tcl
# 或 GUI：Tools → Run Tcl Script...
# ============================================================================
set proj_dir [file normalize "vivado/isar_rtl"]
set part xczu19eg-ffvc1760-2-i

create_project isar_rtl $proj_dir -part $part

# 自研 RTL
add_files -fileset sources_1 [file normalize "src/rtl/pw_conv_engine.v"]
add_files -fileset sources_1 [file normalize "src/rtl/dw_conv_engine.v"]

# 出厂约束（LED/按键/UART 等，按需 add 到 constrs_1）
add_files -fileset constrs_1 [file normalize "refs/factory_vivado/board_test.srcs/constrs_1/new/gpio.xdc"]
add_files -fileset constrs_1 [file normalize "refs/factory_vivado/board_test.srcs/constrs_1/new/uart.xdc"]
add_files -fileset constrs_1 [file normalize "refs/factory_vivado/board_test.srcs/constrs_1/new/system.xdc"]

# 出厂自定义 IP 仓库（extender/frequency/i2c_extender/pwm 等，按需使用）
set_property ip_repo_paths [list [file normalize "refs/factory_vivado/ip_repo"]] [current_fileset]
update_ip_catalog

puts "=== 工程已创建: $proj_dir ==="
puts "下一步：按 docs/platform_bringup.md 在 GUI 里加 CIPS + DDR4 并生成 bitstream"
