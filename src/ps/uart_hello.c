// ============================================================================
// uart_hello.c — PS 侧最小裸机程序（Vitis standalone BSP）
// 用途：W1 平台闭环第二步，验证 PS 启动 + UART0 输出正常。
// 编译：Vitis 建 Application Project，选导出好的 .xsa 平台（见 platform_bringup.md）
// 依赖 BSP：standalone，外设 UART0（默认 -D__clang__ 无关，纯 C99）
// ============================================================================
#include <stdio.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "xil_types.h"

int main(void)
{
    xil_printf("\r\n===== ISAR W1 platform bring-up =====\r\n");
    xil_printf("Device : ZU19EG (ALINX Z19)\r\n");
    xil_printf("UART0  : ready\r\n");

    u32 cnt = 0;
    while (1) {
        xil_printf("tick %lu\r\n", (unsigned long)cnt++);
        // 简易延时（约 1 秒 @525MHz，粗算即可）
        for (volatile u32 i = 0; i < 200000000; i++);
    }
    return 0;
}
