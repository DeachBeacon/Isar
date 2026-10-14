// ============================================================================
// dma_loopback.c — PS 侧 AXI-DMA 环回（Vitis standalone BSP）
// 用途：W1 平台闭环关键一步，验证 PL↔PS 的 DMA 数据通路。
// 参考：course_s1/24_dma_loopback。依赖 BSP：xaxidma + 中断（可不使能）
// ============================================================================
#include <stdio.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_types.h"

#define DMA_DEV_ID   XPAR_AXIDMA_0_DEVICE_ID   // 按自己 .xsa 改
#define BUF_LEN      (256 * 256 * 3)            // 与一张图一致

static u8 tx_buf[BUF_LEN] __attribute__((aligned(64)));
static u8 rx_buf[BUF_LEN] __attribute__((aligned(64)));

static XAxiDma dma;

int dma_init(void)
{
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!cfg) { xil_printf("DMA lookup fail\r\n"); return 1; }
    if (XAxiDma_CfgInitialize(&dma, cfg) != XST_SUCCESS) { xil_printf("DMA init fail\r\n"); return 1; }
    XAxiDma_Reset(&dma);
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK);
    return 0;
}

int main(void)
{
    xil_printf("\r\nDMA loopback test\r\n");
    if (dma_init()) return 1;

    for (u32 i = 0; i < BUF_LEN; i++) tx_buf[i] = (u8)(i & 0xFF);

    // 注意：DMA 读的是物理地址（裸机下 = 虚拟地址，但必须是 cache 对齐的连续内存）
    Xil_DCacheFlushRange((UINTPTR)tx_buf, BUF_LEN);
    Xil_DCacheInvalidateRange((UINTPTR)rx_buf, BUF_LEN);

    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)tx_buf, BUF_LEN, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
        xil_printf("TX start fail\r\n"); return 1;
    }
    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)rx_buf, BUF_LEN, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
        xil_printf("RX start fail\r\n"); return 1;
    }

    while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) ;
    while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA)) ;

    Xil_DCacheInvalidateRange((UINTPTR)rx_buf, BUF_LEN);

    u32 err = 0;
    for (u32 i = 0; i < BUF_LEN; i++) if (rx_buf[i] != tx_buf[i]) err++;
    if (err == 0) xil_printf("loopback OK: %d bytes identical\r\n", BUF_LEN);
    else          xil_printf("loopback FAIL: %lu mismatches\r\n", (unsigned long)err);

    return 0;
}
