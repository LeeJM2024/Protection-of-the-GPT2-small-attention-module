# Protection-of-the-GPT2-small-attention-module
A GPT-2 inference repository built on a simple and insufficiently optimized OP-TEE environment, implementing protection for one of its attention modules.

**[[README_en]](README_en.md)**

## Getting Started
Please refer to my teammate’s implementation first, since my work is an extension based on it:
https://github.com/internetsb/gptz.c

## New Additions in This Implementation
1. The TA heap memory needs to be expanded again.
```c
/* 1. Modify optee_os/core/mm/pgt_cache.c */
// #define PGT_CACHE_SIZE ROUNDUP(CFG_PGT_CACHE_ENTRIES, PGT_NUM_PGT_PER_PAGE)
#define PGT_CACHE_SIZE 512

/* 2. Modify trusted-firmware-a/plat/qemu/qemu/include/platform_def.h */
#define SEC_DRAM_BASE           0x70000000   // Base address
#define SEC_DRAM_SIZE           0x0DC00000   // Size: 220 MB

/* 3. Add one line to optee_os/core/arch/arm/plat-vexpress/platform_config.h */
#define TEE_RAM_VA_SIZE (220 * 1024 * 1024) // 220 MB

/* 4. Modify the qemu_virt configuration in optee_os/core/arch/arm/plat-vexpress/conf.mk */
CFG_TZDRAM_START ?= 0x70000000 # Does not conflict with Kernel (0x42200000)
CFG_TZDRAM_SIZE  ?= 0x0DC00000 # 220 MB

/* 5. Add one line at the end of build/qemu_v8.mk */
OPTEE_OS_COMMON_FLAGS += CFG_TZDRAM_START=0x70000000 CFG_TZDRAM_SIZE=0x0DC00000

/* 6. Modify optee_examples/gpt/ta/user_ta_header_defines.h */
#define TA_DATA_SIZE            (200 * 1024 * 1024) // 200 MB heap memory
```

2. Run the program.
```bash
gpt /mnt/shared/models/gpt2_124M.bin /mnt/shared/models/gpt2_ranks.bin -T 2 [-P <layer_id>]
```
Move the Embedding layer, one Transformer block, and the final LayerNorm and Softmax into the TA. You will then see output similar to the following in the Normal World:
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

 Text to complete: Ladies and # text to be completed
 # model output
 Generated:  Gentlemen, this year's NBA playoffs have never been about making your living from Twitter. But this year is about living with it. We
 #......
```
You can modify the `coin` parameter in the `sample_mult` function in `host/main.c` to change the randomness of the generated text.

## Command-Line Arguments
`-T 0`: All parameters are inferred in the Normal World.
`-T 1`: Protect the Embedding layer parameters.
`-T 2 -P <layer_id>`:
1. Protect the Embedding layer parameters.
2. Support protection of one specified Transformer block.
   Use `-P <block_id>` to specify the protected layer (for example, `-P 0`). For that layer, only the target layer slices of `ln1/qkv/attproj/ln2/fc/fcproj` parameters are sent into the TA, and the forward pass of that layer is executed continuously inside the TA, returning only the final `residual3` output.
3. Execute the final LayerNorm and Softmax inside the TA.
   Under `-T 2`, `lnfw/lnfb` are loaded into the TA, and the last normalization layer and Softmax are completed inside the TA before outputting the result.

**Protecting Large Models with OP-TEE**
This project implements protection for the Transformer architecture and secures all layer structures within a Transformer block, including the LN1 normalization layer, QKV linear layer, Attention computation layer, Attention output projection layer, Residual2 layer, LN2 normalization layer, FC fully connected layer, GELU activation layer, FCProj fully connected output projection layer, and Residual3 layer. It also protects the Token Embedding parameters, as well as the final LayerNorm and Softmax, essentially covering the full GPT-2 small inference pipeline.

However, due to the limited TEE memory in OP-TEE, the current implementation can only protect one block at a time. Since GPT-2 small contains 12 Transformer blocks in total, the security of the entire model inference process cannot yet be fully guaranteed.

## References
- [llm.c](https://github.com/karpathy/llm.c)
- [darknetz](https://github.com/mofanv/darknetz)
- [gpt.c](https://git.nju.edu.cn/jyy/os2025/-/tree/M6)
- [gptz.c](https://github.com/internetsb/gptz.c)
