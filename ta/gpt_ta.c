#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <gpt_ta.h>
#include <model_ta.h>

// 参数和结果空间结构指针
ParameterTensors_TA param_tensors;
ActivationTensors_TA act_tensors;
// 分块加载参数静态变量
static float *full_params_ptr = NULL;
static size_t total_params_size = 0;
static uint32_t wte_size_global = 0;
// 有关注意力block的全局缓存
static float *secure_qkvw = NULL;
static float *secure_qkvb = NULL;
static int secure_C = 0;
static int secure_layer = -1;
static float *secure_ln1w = NULL;
static float *secure_ln1b = NULL;
static float *secure_attprojw = NULL;
static float *secure_attprojb = NULL;
static float *secure_ln2w = NULL;
static float *secure_ln2b = NULL;
static float *secure_fcw = NULL;
static float *secure_fcb = NULL;
static float *secure_fcprojw = NULL;
static float *secure_fcprojb = NULL;
static int secure_NH = 0;
// ------------------------ TA 基本函数--------------------------

/* 当TA实例被创建时调用。这是TA的第一个调用 */
TEE_Result TA_CreateEntryPoint(void)
{
	DMSG("has been called");

	return TEE_SUCCESS;
}

/* 当TA实例被销毁时调用。这是TA的最后一个调用，前提是TA没有崩溃。 */
void TA_DestroyEntryPoint(void)
{
	DMSG("has been called");
}

/* 
 * 当一个新的会话被打开到TA时调用。
 * sess_ctx可以更新为能够识别此会话的会话上下文。在这个函数中，您通常会进行TA的全局初始化。 
 */
TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
				    TEE_Param __unused params[4],
				    void __unused **sess_ctx)
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	IMSG("Initialized\n");

	/* 如果返回值不等于TEE_SUCCESS，则不会创建会话。*/
	return TEE_SUCCESS;
}

/* 当一个会话被关闭时调用。sess_ctx 值是 TA_OpenSessionEntryPoint() 中设置的 */
void TA_CloseSessionEntryPoint(void __unused *sess_ctx) {
    if (full_params_ptr) { TEE_Free(full_params_ptr); full_params_ptr = NULL; }
    if (param_tensors.lnfw) { TEE_Free(param_tensors.lnfw); param_tensors.lnfw = NULL; }
    if (act_tensors.encoded) { TEE_Free(act_tensors.encoded); act_tensors.encoded = NULL; }
    if (act_tensors.logits) { TEE_Free(act_tensors.logits); act_tensors.logits = NULL; }
    if (act_tensors.probs) { TEE_Free(act_tensors.probs); act_tensors.probs = NULL; }
    if (act_tensors.lnf) { TEE_Free(act_tensors.lnf); act_tensors.lnf = NULL; }
	if (secure_qkvw) { TEE_Free(secure_qkvw); secure_qkvw = NULL; }
	if (secure_qkvb) { TEE_Free(secure_qkvb); secure_qkvb = NULL; }
	if (secure_ln1w) { TEE_Free(secure_ln1w); secure_ln1w = NULL; }
	if (secure_ln1b) { TEE_Free(secure_ln1b); secure_ln1b = NULL; }
	if (secure_attprojw) { TEE_Free(secure_attprojw); secure_attprojw = NULL; }
	if (secure_attprojb) { TEE_Free(secure_attprojb); secure_attprojb = NULL; }
	if (secure_ln2w) { TEE_Free(secure_ln2w); secure_ln2w = NULL; }
	if (secure_ln2b) { TEE_Free(secure_ln2b); secure_ln2b = NULL; }
	if (secure_fcw) { TEE_Free(secure_fcw); secure_fcw = NULL; }
	if (secure_fcb) { TEE_Free(secure_fcb); secure_fcb = NULL; }
	if (secure_fcprojw) { TEE_Free(secure_fcprojw); secure_fcprojw = NULL; }
	if (secure_fcprojb) { TEE_Free(secure_fcprojb); secure_fcprojb = NULL; }
	if (act_tensors.ln1) { TEE_Free(act_tensors.ln1); act_tensors.ln1 = NULL; }
	if (act_tensors.qkv) { TEE_Free(act_tensors.qkv); act_tensors.qkv = NULL; }
	if (act_tensors.atty) { TEE_Free(act_tensors.atty); act_tensors.atty = NULL; }
	if (act_tensors.attproj) { TEE_Free(act_tensors.attproj); act_tensors.attproj = NULL; }
	if (act_tensors.residual2) { TEE_Free(act_tensors.residual2); act_tensors.residual2 = NULL; }
	if (act_tensors.ln2) { TEE_Free(act_tensors.ln2); act_tensors.ln2 = NULL; }
	if (act_tensors.fch) { TEE_Free(act_tensors.fch); act_tensors.fch = NULL; }
	if (act_tensors.fch_gelu) { TEE_Free(act_tensors.fch_gelu); act_tensors.fch_gelu = NULL; }
	if (act_tensors.fcproj) { TEE_Free(act_tensors.fcproj); act_tensors.fcproj = NULL; }
	if (act_tensors.residual3) { TEE_Free(act_tensors.residual3); act_tensors.residual3 = NULL; }
	secure_C = 0;
	secure_NH = 0;
	secure_layer = -1;

    IMSG("Resources freed, Session closed.");
}
// ------------------------ TA 函数实现 --------------------------

/**
 * @brief 矩阵乘法前向传播（单线程）
 * @param[out] out 输出张量，形状为(B, T, OC)
 * @param inp 输入张量，形状为(B, T, C)
 * @param weight 权重矩阵，形状为(C, OC)
 * @param bias 偏置向量，形状为(OC)，可为NULL
 * @param B 批次大小
 * @param T 序列长度
 * @param C 输入通道数
 * @param OC 输出通道数
 */
static void matmul_forward_TA(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o * C;
                
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                
                out_bt[o] = val;
            }
        }
    }
}

/**
 * @brief 在词汇表维度上执行softmax操作
 * @param[out] probs 输出概率，形状为(B, T, V)
 * @param logits 输入逻辑值，形状为(B, T, V)
 * @param B 批次大小
 * @param T 序列长度
 * @param V 词汇表大小
 */
static void softmax_forward_TA(float* probs, float* logits, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            float maxval = -10000.0f;
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) maxval = logits_bt[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = (float)ta_exp((double)(logits_bt[i] - maxval));
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

/**
 * @brief LayerNorm 前向传播 (TA内部实现)
 */
static void layernorm_forward_TA_impl(float* out, float* inp, float* weight, float* bias, int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) m += x[i];
            m = m / C;
            
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / (float)ta_sqrt((double)(v + eps));
            
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = (s * (x[i] - m)) * weight[i] + bias[i];
            }
        }
    }
}

/**
 * @brief Attention 前向传播 (TA内部实现)
 */
static void attention_forward_TA_impl(float* out, float* inp, int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / (float)ta_sqrt((double)hs);
    float* att_tmp = TEE_Malloc((size_t)T * sizeof(float), 0);
    if (!att_tmp) return;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float maxval = -10000.0f;

                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
                    val *= scale;
                    if (val > maxval) maxval = val;
                    att_tmp[t2] = val;
                }

                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = (float)ta_exp((double)(att_tmp[t2] - maxval));
                    expsum += expv;
                    att_tmp[t2] = expv;
                }
                float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;

                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float a = att_tmp[t2] * inv;
                    for (int i = 0; i < hs; i++) out_bth[i] += a * value_t2[i];
                }
            }
        }
    }
    TEE_Free(att_tmp);
}

static float tanh_from_exp(float x) {
    double e2x = ta_exp((double)(2.0f * x));
    return (float)((e2x - 1.0) / (e2x + 1.0));
}

/**
 * @brief GELU 前向传播 (TA内部实现)
 */
static void gelu_forward_TA_impl(float* out, float* inp, int N) {
    const float k = 0.7978845608f; // sqrt(2/pi)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanh_from_exp(k * (x + cube)));
    }
}

/**
 * @brief Residual 前向传播 (TA内部实现)
 */
static void residual_forward_TA_impl(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) out[i] = inp1[i] + inp2[i];
}

// -------------------- TA 命令入口 --------------------------
/**
 * @brief 加载LayerNorm的权重和偏置
 */
static TEE_Result load_lnfwb_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
					      TEE_PARAM_TYPE_MEMREF_INPUT,
					      TEE_PARAM_TYPE_NONE,
					      TEE_PARAM_TYPE_NONE);
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	
	param_tensors.lnfw = TEE_Malloc(params[0].memref.size, 0);
	TEE_MemMove(param_tensors.lnfw, params[0].memref.buffer, params[0].memref.size);
	param_tensors.lnfb = TEE_Malloc(params[1].memref.size, 0);
	TEE_MemMove(param_tensors.lnfb, params[1].memref.buffer, params[1].memref.size);

	IMSG("[TA] load_lnfwb: lnfw_bytes=%u lnfb_bytes=%u", (unsigned)params[0].memref.size, (unsigned)params[1].memref.size);
	return TEE_SUCCESS;
}

/**
 * @brief 分块加载模型参数wte,wpe到TEE内存
 */
static TEE_Result load_parameters_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, 0, 0);
    uint32_t init_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, 0, 0);

    if (param_types == init_pt) {
        size_t req_sz = params[0].value.a;
        wte_size_global = params[0].value.b;
        if (full_params_ptr) TEE_Free(full_params_ptr);
        full_params_ptr = TEE_Malloc(req_sz, TEE_MALLOC_FILL_ZERO);
        if (!full_params_ptr) return TEE_ERROR_OUT_OF_MEMORY;
        total_params_size = req_sz;
        return TEE_SUCCESS;
    }

    if (param_types == exp_pt) {
        if (!full_params_ptr) return TEE_ERROR_BAD_STATE;
        uint32_t offset = params[1].value.a;
        size_t sz = params[0].memref.size;
        if (offset + sz > total_params_size) return TEE_ERROR_BAD_PARAMETERS;
        
        TEE_MemMove((uint8_t*)full_params_ptr + offset, params[0].memref.buffer, sz);

        if (offset + sz == total_params_size) {
            param_tensors.wte = full_params_ptr;
            param_tensors.wpe = full_params_ptr + wte_size_global;
        }
        return TEE_SUCCESS;
    }
    return TEE_ERROR_BAD_PARAMETERS;
}

/**
 * @brief 加载注意力block的指定层模型到TEE内存
 */
static TEE_Result load_block_tensor_TA(float **dst, uint32_t param_types, TEE_Param params[4], const char* name) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                   TEE_PARAM_TYPE_VALUE_INPUT,
                                   TEE_PARAM_TYPE_NONE,
                                   TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int C = (int)params[1].value.a;
    int layer_id = (int)params[1].value.b;
    size_t bytes = params[0].memref.size;

    if (*dst) { TEE_Free(*dst); *dst = NULL; }
    *dst = TEE_Malloc(bytes, 0);
    if (!*dst) return TEE_ERROR_OUT_OF_MEMORY;

    TEE_MemMove(*dst, params[0].memref.buffer, bytes);
    secure_C = C;
    secure_layer = layer_id;
    IMSG("[TA] loaded %s layer=%d C=%d bytes=%lu", name, layer_id, C, (unsigned long)bytes);
    return TEE_SUCCESS;
}

/**
 * @brief 加载qkvw和qkvb
 */
static TEE_Result load_qkvw_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_qkvw, param_types, params, "qkvw"); }
static TEE_Result load_qkvb_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_qkvb, param_types, params, "qkvb"); }
/**
 * @brief 加载ln1w和ln1b
 */
static TEE_Result load_ln1w_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_ln1w, param_types, params, "ln1w"); }
static TEE_Result load_ln1b_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_ln1b, param_types, params, "ln1b"); }
/**
 * @brief 加载attprojw和attprojb
 */
static TEE_Result load_attprojw_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_attprojw, param_types, params, "attprojw"); }
static TEE_Result load_attprojb_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_attprojb, param_types, params, "attprojb"); }
/**
 * @brief 加载ln2w和ln2b
 */
static TEE_Result load_ln2w_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_ln2w, param_types, params, "ln2w"); }
static TEE_Result load_ln2b_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_ln2b, param_types, params, "ln2b"); }
/**
 * @brief 加载fcw和fcb
 */
static TEE_Result load_fcw_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_fcw, param_types, params, "fcw"); }
static TEE_Result load_fcb_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_fcb, param_types, params, "fcb"); }
/**
 * @brief 加载fcprojw和fcprojb
 */
static TEE_Result load_fcprojw_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_fcprojw, param_types, params, "fcprojw"); }
static TEE_Result load_fcprojb_TA(uint32_t param_types, TEE_Param params[4]) { return load_block_tensor_TA(&secure_fcprojb, param_types, params, "fcprojb"); }


/**
 * @brief 输出受保护block的最终结果（residual3层的输出）
 */
static TEE_Result block_output_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                   TEE_PARAM_TYPE_NONE,
                                   TEE_PARAM_TYPE_NONE,
                                   TEE_PARAM_TYPE_NONE);
    if (param_types != exp || !act_tensors.residual3)
        return TEE_ERROR_BAD_PARAMETERS;
    TEE_MemMove(params[0].memref.buffer, act_tensors.residual3, params[0].memref.size);
    return TEE_SUCCESS;
}

/**
 * @brief 重新分配TA内部张量缓冲区
 */
static TEE_Result realloc_tensor_TA(float **dst, size_t bytes) {
    if (*dst) { TEE_Free(*dst); *dst = NULL; }
    *dst = TEE_Malloc(bytes, 0);
    if (!*dst) return TEE_ERROR_OUT_OF_MEMORY;
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中第一个归一化层ln1前向传播
 */
static TEE_Result ln1_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[1].value.a;
    int T = (int)params[1].value.b;
    int C = (int)params[2].value.a;
    float *inp = (float*)params[0].memref.buffer;

    if (!secure_ln1w || !secure_ln1b || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.ln1, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    layernorm_forward_TA_impl(act_tensors.ln1, inp, secure_ln1w, secure_ln1b, B, T, C);
    IMSG("ln1_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中qkv投影前向传播
 */
static TEE_Result qkv_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.ln1 || !secure_qkvw || !secure_qkvb || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.qkv, (size_t)B * T * (3 * C) * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    matmul_forward_TA(act_tensors.qkv, act_tensors.ln1, secure_qkvw, secure_qkvb, B, T, C, 3 * C);
    IMSG("qkv_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中attention层前向传播
 */
static TEE_Result attention_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;
    int NH = (int)params[1].value.b;

    if (!act_tensors.qkv || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.atty, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    secure_NH = NH;
    attention_forward_TA_impl(act_tensors.atty, act_tensors.qkv, B, T, C, NH);
    IMSG("attention_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中注意力投影attproj前向传播
 */
static TEE_Result attproj_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.atty || !secure_attprojw || !secure_attprojb || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.attproj, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    matmul_forward_TA(act_tensors.attproj, act_tensors.atty, secure_attprojw, secure_attprojb, B, T, C, C);
    IMSG("attproj_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中残差层residual2前向传播
 */
static TEE_Result residual2_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[1].value.a;
    int T = (int)params[1].value.b;
    int C = (int)params[2].value.a;
    float *residual = (float*)params[0].memref.buffer;

    if (!act_tensors.attproj || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.residual2, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    residual_forward_TA_impl(act_tensors.residual2, residual, act_tensors.attproj, B * T * C);
    IMSG("residual2_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中第二个归一化层ln2前向传播
 */
static TEE_Result ln2_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.residual2 || !secure_ln2w || !secure_ln2b || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.ln2, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    layernorm_forward_TA_impl(act_tensors.ln2, act_tensors.residual2, secure_ln2w, secure_ln2b, B, T, C);
    IMSG("ln2_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中FNN第一层全连接fc前向传播
 */
static TEE_Result fc_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.ln2 || !secure_fcw || !secure_fcb || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.fch, (size_t)B * T * (4 * C) * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    matmul_forward_TA(act_tensors.fch, act_tensors.ln2, secure_fcw, secure_fcb, B, T, C, 4 * C);
    IMSG("fc_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中激活层gelu前向传播
 */
static TEE_Result gelu_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.fch || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.fch_gelu, (size_t)B * T * (4 * C) * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    gelu_forward_TA_impl(act_tensors.fch_gelu, act_tensors.fch, B * T * 4 * C);
    IMSG("gelu_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中FNN投影层fcproj前向传播
 */
static TEE_Result fcproj_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.fch_gelu || !secure_fcprojw || !secure_fcprojb || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.fcproj, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    matmul_forward_TA(act_tensors.fcproj, act_tensors.fch_gelu, secure_fcprojw, secure_fcprojb, B, T, 4 * C, C);
    IMSG("fcproj_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 保护块中残差层residual3前向传播
 */
static TEE_Result residual3_forward_TA(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    int B = (int)params[0].value.a;
    int T = (int)params[0].value.b;
    int C = (int)params[1].value.a;

    if (!act_tensors.residual2 || !act_tensors.fcproj || C != secure_C) return TEE_ERROR_BAD_STATE;
    if (realloc_tensor_TA(&act_tensors.residual3, (size_t)B * T * C * sizeof(float)) != TEE_SUCCESS) return TEE_ERROR_OUT_OF_MEMORY;

    residual_forward_TA_impl(act_tensors.residual3, act_tensors.residual2, act_tensors.fcproj, B * T * C);
    IMSG("residual3_forward layer=%d", secure_layer);
    return TEE_SUCCESS;
}

/**
 * @brief 编码器前向传播：token嵌入 + 位置嵌入
 */
static TEE_Result encoder_forward_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	// 解析参数
	int* inputs = (int*)params[0].memref.buffer;
	int B = params[1].value.a;
	int T = params[1].value.b;
	int C = params[2].value.a;
    // 分配输出内存
	int output_size = B * T * C * sizeof(float);
	if (act_tensors.encoded) TEE_Free(act_tensors.encoded);
    act_tensors.encoded = TEE_Malloc(output_size, 0);
	if (!act_tensors.encoded) {
		EMSG("Out of memory");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	// 计算
	float* out = act_tensors.encoded;
	float* wte = param_tensors.wte;
	float* wpe = param_tensors.wpe;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inputs[b * T + t];
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
	return TEE_SUCCESS;
}

/**
 * @brief 获取编码器输出
 */
static TEE_Result encoder_output_TA(uint32_t param_types, TEE_Param params[4])
{ 
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 输出act_tensors.encoded到输出缓冲区
	float *buffer = (float*)params[0].memref.buffer;
	size_t buffer_size = params[0].memref.size;
	TEE_MemMove(buffer, act_tensors.encoded, buffer_size);
	return TEE_SUCCESS;
}
// matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
// softmax_forward(acts.probs, acts.logits, B, T, V);
/**
 * @brief 矩阵乘法和Softmax前向传播
 */
static TEE_Result matmul_softmax_forward_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 解析参数
	float* inputs = (float*)params[0].memref.buffer;
	int B = params[1].value.a;
	int T = params[1].value.b;
	int C = params[2].value.a;
	int V = params[2].value.b;
	// 分配输出内存
	int output_size = B * T * V * sizeof(float);
	if (act_tensors.logits) { TEE_Free(act_tensors.logits); act_tensors.logits = NULL; }
    if (act_tensors.probs) { TEE_Free(act_tensors.probs); act_tensors.probs = NULL; }

    act_tensors.logits = TEE_Malloc(output_size, 0);
    act_tensors.probs = TEE_Malloc(output_size, 0);
	if (!act_tensors.logits || !act_tensors.probs) {
		EMSG("Out of memory");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	// 计算
	matmul_forward_TA(act_tensors.logits, inputs, param_tensors.wte, NULL, B, T, C, V);
	softmax_forward_TA(act_tensors.probs, act_tensors.logits, B, T, V);
	return TEE_SUCCESS;
}

/**
 * @brief 获取Softmax输出
 */
static TEE_Result softmax_output_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 输出act_tensors.probs到输出缓冲区
	float *buffer = (float*)params[0].memref.buffer;
	size_t buffer_size = params[0].memref.size;
	TEE_MemMove(buffer, act_tensors.probs, buffer_size);
	return TEE_SUCCESS;
}

/**
 * @brief LayerNorm前向传播
 */
static TEE_Result layernorm_forward_TA(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_VALUE_INPUT,
                                   TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) return TEE_ERROR_BAD_PARAMETERS;

    float* inp = (float*)params[0].memref.buffer;
    int B = params[1].value.a;
    int T = params[1].value.b;
    int C = params[2].value.a;

    size_t out_size = B * T * C * sizeof(float);
    if (act_tensors.lnf) { TEE_Free(act_tensors.lnf); act_tensors.lnf = NULL; }
    act_tensors.lnf = TEE_Malloc(out_size, 0);
    if (!act_tensors.lnf) return TEE_ERROR_OUT_OF_MEMORY;

    layernorm_forward_TA_impl(act_tensors.lnf, inp, param_tensors.lnfw, param_tensors.lnfb, B, T, C);
    return TEE_SUCCESS;
}

/**
 * @brief 返回LayerNorm输出
 */
static TEE_Result layernorm_output_TA(uint32_t param_types, TEE_Param params[4])
{
    if (param_types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE))
        return TEE_ERROR_BAD_PARAMETERS;
    
    TEE_MemMove(params[0].memref.buffer, act_tensors.lnf, params[0].memref.size);
    return TEE_SUCCESS;
}

/* 当一个TA被调用时调用。sess_ctx保存由TA_OpenSessionEntryPoint()设置的值。其余的参数来自普通世界 */
TEE_Result TA_InvokeCommandEntryPoint(void __unused *sess_ctx,
				      uint32_t cmd_id, uint32_t param_types,
				      TEE_Param params[4])
{
	switch (cmd_id) {
	case TA_GPT_CMD_LOAD_PARAMS:
		return load_parameters_TA(param_types, params);
	case TA_GPT_CMD_ENCODER_FORWARD:
		return encoder_forward_TA(param_types, params);
	case TA_GPT_CMD_ENCODER_OUTPUT:
		return encoder_output_TA(param_types, params);
	case TA_GPT_CMD_SOFTMAX_FORWARD:
		return matmul_softmax_forward_TA(param_types, params);
	case TA_GPT_CMD_SOFTMAX_OUTPUT:
		return softmax_output_TA(param_types, params);
	case TA_GPT_CMD_LOAD_LNFWB:
		return load_lnfwb_TA(param_types, params);
	case TA_GPT_CMD_LAYERNORM_FORWARD:
		return layernorm_forward_TA(param_types, params);
	case TA_GPT_CMD_LAYERNORM_OUTPUT:
		return layernorm_output_TA(param_types, params);

	case TA_GPT_CMD_LOAD_LN1W:
		return load_ln1w_TA(param_types, params);
	case TA_GPT_CMD_LOAD_LN1B:
		return load_ln1b_TA(param_types, params);
    case TA_GPT_CMD_LOAD_QKVW:
    	return load_qkvw_TA(param_types, params);
	case TA_GPT_CMD_LOAD_QKVB:
    	return load_qkvb_TA(param_types, params);
	case TA_GPT_CMD_LOAD_ATTPROJW:
		return load_attprojw_TA(param_types, params);
	case TA_GPT_CMD_LOAD_ATTPROJB:
		return load_attprojb_TA(param_types, params);
	case TA_GPT_CMD_LOAD_LN2W:
		return load_ln2w_TA(param_types, params);
	case TA_GPT_CMD_LOAD_LN2B:
		return load_ln2b_TA(param_types, params);
	case TA_GPT_CMD_LOAD_FCW:
		return load_fcw_TA(param_types, params);
	case TA_GPT_CMD_LOAD_FCB:
		return load_fcb_TA(param_types, params);
	case TA_GPT_CMD_LOAD_FCPROJW:
		return load_fcprojw_TA(param_types, params);
	case TA_GPT_CMD_LOAD_FCPROJB:
		return load_fcprojb_TA(param_types, params);

	case TA_GPT_CMD_LN1_FORWARD:
		return ln1_forward_TA(param_types, params);
	case TA_GPT_CMD_QKV_FORWARD:
		return qkv_forward_TA(param_types, params);
	case TA_GPT_CMD_ATTENTION_FORWARD:
		return attention_forward_TA(param_types, params);
	case TA_GPT_CMD_ATTPROJ_FORWARD:
		return attproj_forward_TA(param_types, params);
	case TA_GPT_CMD_RESIDUAL2_FORWARD:
		return residual2_forward_TA(param_types, params);
	case TA_GPT_CMD_LN2_FORWARD:
		return ln2_forward_TA(param_types, params);
	case TA_GPT_CMD_FC_FORWARD:
		return fc_forward_TA(param_types, params);
	case TA_GPT_CMD_GELU_FORWARD:
		return gelu_forward_TA(param_types, params);
	case TA_GPT_CMD_FCPROJ_FORWARD:
		return fcproj_forward_TA(param_types, params);
	case TA_GPT_CMD_RESIDUAL3_FORWARD:
		return residual3_forward_TA(param_types, params);
	case TA_GPT_CMD_BLOCK_OUTPUT:
		return block_output_TA(param_types, params);

	default:
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
