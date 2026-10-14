// ============================================================================
// sd_read_image.c — SD 卡读一张 256x256x3 图到 DDR（Vitis standalone BSP）
// 用途：W1 平台闭环第三步，验证"数据能进 PS"。
// 依赖 BSP：xilffs（FATFS）、xilsecure/xilpm（SD 初始化）、xil_cache
// 编译：Application Project 里勾选 xilffs + xilsecure + xilpm 库
// ============================================================================
#include <stdio.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "ff.h"                 // xilffs
#include "xil_cache.h"
#include "xil_types.h"

#define IMG_W     256
#define IMG_H     256
#define IMG_CH    3
#define IMG_BYTES (IMG_W * IMG_H * IMG_CH)   // 196608 = 192KB

// 期望文件：SD 卡 FAT32 分区根目录的 input.bin（BGR/RGB 逐字节，CHW 或 HWC 见注释）
#define IMG_PATH  "input.bin"

static u8  img_buf[IMG_BYTES] __attribute__((aligned(64)));

int main(void)
{
    Xil_DCacheDisable();   // 简化缓存一致性；后续用 DMA 时改成 flush/invalidate

    FATFS fs;
    FIL   fil;
    FRESULT fr;
    UINT  br = 0;
    TCHAR *path = IMG_PATH;

    xil_printf("\r\nSD read test\r\n");

    fr = f_mount(&fs, path, 0);          // 挂载
    if (fr != FR_OK) {
        xil_printf("f_mount fail: %d\r\n", fr);
        return 1;
    }
    fr = f_open(&fil, path, FA_READ);
    if (fr != FR_OK) {
        xil_printf("f_open fail: %d (请确认 SD 卡根目录有 %s)\r\n", fr, path);
        return 1;
    }
    fr = f_read(&fil, img_buf, IMG_BYTES, &br);
    if (fr != FR_OK || br != IMG_BYTES) {
        xil_printf("f_read fail: %d, read %lu / %d\r\n", fr, (unsigned long)br, IMG_BYTES);
        f_close(&fil);
        return 1;
    }
    f_close(&fil);
    f_mount(0, path, 0);

    // 校验和回报
    u32 sum = 0;
    for (u32 i = 0; i < IMG_BYTES; i++) sum += img_buf[i];
    xil_printf("read OK, bytes=%d, checksum=0x%08lx\r\n", IMG_BYTES, (unsigned long)sum);

    // TODO(W2): 把 img_buf 首地址交给 DMA，送到 PL 卷积引擎
    return 0;
}
