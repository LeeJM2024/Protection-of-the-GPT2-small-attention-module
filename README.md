# Protection-of-the-GPT2-small-attention-module
基于一个简单、未充分优化的optee环境的gpt2推理仓库，实现对其一个注意力模块的保护

**[[README_en]](README_en.md)**

## 如何开始？
参考同学的实现(因为我是基于他的实现进行补充的)
https://github.com/internetsb/gptz.c

## 以下为新实现内容
1. 需再次扩展TA堆内存
```
/* 1.修改optee_os/core/mm/pgt_cache.c */
// #define PGT_CACHE_SIZE	ROUNDUP(CFG_PGT_CACHE_ENTRIES, PGT_NUM_PGT_PER_PAGE)
#define PGT_CACHE_SIZE 512

/* 2.修改trusted-firmware-a/plat/qemu/qemu/include/platform_def.h */
#define SEC_DRAM_BASE           0x70000000   // 基地址
#define SEC_DRAM_SIZE           0x0DC00000	 // 大小 220MB

/* 3.在optee_os/core/arch/arm/plat-vexpress/platform_config.h增加一行 */
#define TEE_RAM_VA_SIZE (220 * 1024 * 1024) // 220MB

/* 4.修改optee_os/core/arch/arm/plat-vexpress/conf.mk中qemu_virt的配置 */
CFG_TZDRAM_START ?= 0x70000000 # 不与 Kernel (0x42200000) 冲突
CFG_TZDRAM_SIZE  ?= 0x0DC00000 # 220M

/* 5.在build/qemu_v8.mk文件末尾增加一行 */
OPTEE_OS_COMMON_FLAGS += CFG_TZDRAM_START=0x70000000 CFG_TZDRAM_SIZE=0x0DC00000

/* 6.修改optee_examples/gpt/ta/user_ta_header_defines.h */
#define TA_DATA_SIZE			(200 * 1024 * 1024) // 200M堆内存
```

2. 运行程序
  ```bash
  gpt /mnt/shared/models/gpt2_124M.bin /mnt/shared/models/gpt2_ranks.bin -T 2 [-P <layer_id>]
  ```
  将Embedding层、一个Transformer block、以及最后的layernorm和softmax等放入TA，你将看到Normal world出现类似于以下的输出：
  ```
  Session started.
   Parameter 0: 154389504 bytes
   Parameter 1: 3145728 bytes
   #...
   Allocated 497759232 bytes for model parameters
   Uploading parameters: 100.0%
   Done.
   Loaded parameters into TEE...
   [CA] ln1w split: layer=0 pre=0 protected=768 post=8448
   Loaded parameter 2...
   [CA] ln1b split: layer=0 pre=0 protected=768 post=8448
   Loaded parameter 3...
   #...
   [CA] load_lnfwb: lnfw=768 floats, lnfb=768 floats
   Loaded parameters into TEE...
   Loaded 497759232 bytes of model parameters
   Model loaded from: /mnt/shared/models/gpt2_124M.bin
   Tokenizer loaded from: /mnt/shared/models/gpt2_ranks.bin

   Text to complete: Ladies and #待补全文本
   #模型输出
   Generated:  Gentlemen, this year's NBA playoffs have never been about making your living from Twitter. But this year is about living with it. We
   #......
   ```
   你可以修改host/main.c中sample_mult函数的coin参数来改变模型生成文本的随机性。

## 命令行参数
 ```
-T 0 :全部参数在普通世界推理；-T 1:保护Embedding层参数；
-T 2 -P <layer_id>: 
1) 保护Embedding层参数
2) 支持指定保护一个 Transformer block
通过 -P <block_id> 指定受保护层（如 -P 0）。该层的 ln1/qkv/attproj/ln2/fc/fcproj 参数只把目标层切片送入 TA，且该层前向在 TA 内连续执行，仅回传最终 residual3 层输出。
3）最终的 LayerNorm 和 Softmax 在 TA 执行
lnfw/lnfb 在 -T 2 下加载到 TA，最后一层归一化以及 Softmax 在 TA 内完成，再输出结果
 ```

## OP-TEE保护大模型
实现了对 Transformer 架构的保护，保护了一个 Transformer block 中的所有层结构，包括LN1 归一化层、QKV 线性层、Attention 计算层、Attention 输出投影层、Residual2 残差层、LN2 归一化层、FC 全连接层、GELU 激活层、FCProj 全连接输出投影层、Residual3 残差层。保护了 Token Embedding 参数，以及最后的 LayerNorm 和 Softmax，基本实现了对 GPT2-small 的推理全流程保护。
但受限于 OP-TEE 有限的 TEE 内存，目前只能实现对其中一个 block 的保护，而 GPT2-small 中共有12个 Transformer block，所以无法完全保证模型推理的安全。并且在推理一段时间后，程序可能会因内存不足而崩溃。

## 参考
- [llm.c](https://github.com/karpathy/llm.c)
- [darknetz](https://github.com/mofanv/darknetz)
- [gpt.c](https://git.nju.edu.cn/jyy/os2025/-/tree/M6)
- [gptz.c](https://github.com/internetsb/gptz.c)
