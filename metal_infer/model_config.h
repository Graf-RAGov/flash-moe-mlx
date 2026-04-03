/*
 * model_config.h — Compile-time model profile selection for Flash-MoE
 *
 * Supported models:
 *   - MODEL_QWEN35_397B (default): Qwen3.5-397B-A17B-4bit
 *   - MODEL_QWEN3_CODER_480B:      Qwen3-Coder-480B-A35B-Instruct-4bit
 *
 * To select a model, pass -DMODEL_QWEN3_CODER_480B via compiler flags.
 * If neither is defined, MODEL_QWEN35_397B is used as default.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

/* ---- Default model selection ---- */
#if !defined(MODEL_QWEN35_397B) && !defined(MODEL_QWEN3_CODER_480B)
#define MODEL_QWEN35_397B
#endif

/* ======================================================================== */
#if defined(MODEL_QWEN3_CODER_480B)
/* ======================================================================== */
/*
 * Qwen3-Coder-480B-A35B-Instruct-4bit (mlx-community)
 *   - 62 layers, ALL standard full attention (no GatedDeltaNet)
 *   - hidden_size=6144, head_dim=128, 96 Q heads, 8 KV heads (GQA 12:1)
 *   - 160 experts/layer, top_k=8, moe_intermediate=2560
 *   - No shared expert
 *   - Full RoPE on all 128 head dimensions
 *   - No sigmoid gate on attention output
 */

#define MODEL_NAME              "Qwen3-Coder-480B-A35B"
#define MODEL_ID                "qwen3-coder-480b-a35b"

/* Core dimensions */
#define HIDDEN_DIM              6144
#define NUM_LAYERS              62
#define NUM_ATTN_HEADS          96
#define NUM_KV_HEADS            8
#define HEAD_DIM                128
#define VOCAB_SIZE              151936
#define RMS_NORM_EPS            1e-6f

/* MoE */
#define NUM_EXPERTS             160
#define NUM_EXPERTS_PER_TOK     8
#define MOE_INTERMEDIATE        2560
#define SHARED_INTERMEDIATE     0       /* no shared expert */
#define GROUP_SIZE              64
#define BITS                    4

/* Architecture flags */
#define HAS_LINEAR_ATTENTION    0
#define HAS_SHARED_EXPERT       0
#define HAS_ATTN_GATE           0

/* All layers are full attention (every 1st layer = every layer) */
#define FULL_ATTN_INTERVAL      1
#define NUM_FULL_ATTN_LAYERS    62
#define NUM_LINEAR_LAYERS       0

/* Full RoPE (all 128 dims) */
#define ROPE_THETA              10000000.0f
#define PARTIAL_ROTARY          1.0f
#define ROTARY_DIM              128

/*
 * Expert packed binary layout (4-bit, group_size=64)
 *
 * gate_proj: [2560, 6144] → packed [2560, 768] uint32
 * up_proj:   [2560, 6144] → packed [2560, 768] uint32
 * down_proj: [6144, 2560] → packed [6144, 320] uint32
 */
#define EXPERT_SIZE             26542080

#define GATE_W_OFF              0
#define GATE_S_OFF              7864320
#define GATE_B_OFF              8355840
#define UP_W_OFF                8847360
#define UP_S_OFF                16711680
#define UP_B_OFF                17203200
#define DOWN_W_OFF              17694720
#define DOWN_S_OFF              25559040
#define DOWN_B_OFF              26050560

/* Component sizes (for repack_experts.py consistency) */
#define GATE_W_SIZE             7864320     /* [2560, 768] uint32 */
#define GATE_S_SIZE             491520      /* [2560, 96]  bf16   */
#define GATE_B_SIZE             491520
#define UP_W_SIZE               7864320
#define UP_S_SIZE               491520
#define UP_B_SIZE               491520
#define DOWN_W_SIZE             7864320     /* [6144, 320] uint32 */
#define DOWN_S_SIZE             491520      /* [6144, 40]  bf16   */
#define DOWN_B_SIZE             491520

/* 2-bit expert layout (not yet calculated for this model) */
#define EXPERT_SIZE_2BIT        0
#define GATE_W_OFF_2            0
#define GATE_S_OFF_2            0
#define GATE_B_OFF_2            0
#define UP_W_OFF_2              0
#define UP_S_OFF_2              0
#define UP_B_OFF_2              0
#define DOWN_W_OFF_2            0
#define DOWN_S_OFF_2            0
#define DOWN_B_OFF_2            0

/* Special tokens */
#define EOS_TOKEN_1             151645  /* <|im_end|>      */
#define EOS_TOKEN_2             151643  /* <|endoftext|>    */
#define THINK_START_TOKEN       151667  /* <think>          */
#define THINK_END_TOKEN         151668  /* </think>         */

/* KV cache */
#define MAX_SEQ_LEN             1048576
#define GPU_KV_SEQ              8192

/* Default model path (user should set via --model) */
#define MODEL_PATH_DEFAULT      ""

/* Linear attention constants (unused — no GatedDeltaNet) */
#define LINEAR_NUM_V_HEADS      0
#define LINEAR_NUM_K_HEADS      0
#define LINEAR_KEY_DIM          1       /* avoid zero-size arrays */
#define LINEAR_VALUE_DIM        1
#define LINEAR_TOTAL_KEY        0
#define LINEAR_TOTAL_VALUE      0
#define LINEAR_CONV_DIM         0
#define CONV_KERNEL_SIZE        4

/* ======================================================================== */
#elif defined(MODEL_QWEN35_397B)
/* ======================================================================== */
/*
 * Qwen3.5-397B-A17B-4bit (mlx-community)
 *   - 60 layers: 45 linear attention (GatedDeltaNet) + 15 full attention
 *   - hidden_size=4096, head_dim=256, 32 Q heads, 2 KV heads (GQA 16:1)
 *   - 512 experts/layer, top_k=10 (runtime default K=4), moe_intermediate=1024
 *   - Shared expert per layer (always active)
 *   - Partial RoPE (25% of 256 = 64 dims)
 *   - Sigmoid gate on full-attention output
 */

#define MODEL_NAME              "Qwen3.5-397B-A17B"
#define MODEL_ID                "qwen3.5-397b-a17b"

/* Core dimensions */
#define HIDDEN_DIM              4096
#define NUM_LAYERS              60
#define NUM_ATTN_HEADS          32
#define NUM_KV_HEADS            2
#define HEAD_DIM                256
#define VOCAB_SIZE              248320
#define RMS_NORM_EPS            1e-6f

/* MoE */
#define NUM_EXPERTS             512
#define NUM_EXPERTS_PER_TOK     10
#define MOE_INTERMEDIATE        1024
#define SHARED_INTERMEDIATE     1024
#define GROUP_SIZE              64
#define BITS                    4

/* Architecture flags */
#define HAS_LINEAR_ATTENTION    1
#define HAS_SHARED_EXPERT       1
#define HAS_ATTN_GATE           1

/* Full attention every 4th layer */
#define FULL_ATTN_INTERVAL      4
#define NUM_FULL_ATTN_LAYERS    15
#define NUM_LINEAR_LAYERS       45

/* Partial RoPE (25% of HEAD_DIM = 64 dims) */
#define ROPE_THETA              10000000.0f
#define PARTIAL_ROTARY          0.25f
#define ROTARY_DIM              64      /* (int)(HEAD_DIM * PARTIAL_ROTARY) */

/*
 * Expert packed binary layout (4-bit, group_size=64)
 *
 * gate_proj: [1024, 4096] → packed [1024, 512] uint32
 * up_proj:   [1024, 4096] → packed [1024, 512] uint32
 * down_proj: [4096, 1024] → packed [4096, 128] uint32
 */
#define EXPERT_SIZE             7077888

#define GATE_W_OFF              0
#define GATE_S_OFF              2097152
#define GATE_B_OFF              2228224
#define UP_W_OFF                2359296
#define UP_S_OFF                4456448
#define UP_B_OFF                4587520
#define DOWN_W_OFF              4718592
#define DOWN_S_OFF              6815744
#define DOWN_B_OFF              6946816

/* Component sizes */
#define GATE_W_SIZE             2097152     /* [1024, 512] uint32 */
#define GATE_S_SIZE             131072      /* [1024, 64]  bf16   */
#define GATE_B_SIZE             131072
#define UP_W_SIZE               2097152
#define UP_S_SIZE               131072
#define UP_B_SIZE               131072
#define DOWN_W_SIZE             2097152     /* [4096, 128] uint32 */
#define DOWN_S_SIZE             131072      /* [4096, 16]  bf16   */
#define DOWN_B_SIZE             131072

/* 2-bit expert layout */
#define EXPERT_SIZE_2BIT        3932160
#define GATE_W_OFF_2            0
#define GATE_S_OFF_2            1048576
#define GATE_B_OFF_2            1179648
#define UP_W_OFF_2              1310720
#define UP_S_OFF_2              2359296
#define UP_B_OFF_2              2490368
#define DOWN_W_OFF_2            2621440
#define DOWN_S_OFF_2            3670016
#define DOWN_B_OFF_2            3801088

/* Special tokens */
#define EOS_TOKEN_1             248046
#define EOS_TOKEN_2             248044
#define THINK_START_TOKEN       248068  /* <think>  */
#define THINK_END_TOKEN         248069  /* </think> */

/* KV cache */
#define MAX_SEQ_LEN             1048576
#define GPU_KV_SEQ              8192

/* Default model path */
#define MODEL_PATH_DEFAULT      "/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"

/* Linear attention (GatedDeltaNet) constants */
#define LINEAR_NUM_V_HEADS      64
#define LINEAR_NUM_K_HEADS      16
#define LINEAR_KEY_DIM          128
#define LINEAR_VALUE_DIM        128
#define LINEAR_TOTAL_KEY        (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)    /* 2048  */
#define LINEAR_TOTAL_VALUE      (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)  /* 8192  */
#define LINEAR_CONV_DIM         (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) /* 12288 */
#define CONV_KERNEL_SIZE        4

#endif /* model selection */

/* ======================================================================== */
/* Derived constants (shared) */
/* ======================================================================== */

/* Q projection output dimension: doubled if attention has sigmoid gate */
#if HAS_ATTN_GATE
#define Q_PROJ_DIM              (NUM_ATTN_HEADS * HEAD_DIM * 2)
#else
#define Q_PROJ_DIM              (NUM_ATTN_HEADS * HEAD_DIM)
#endif

/* Multi-expert buffer limit */
#define MAX_K                   8

#endif /* MODEL_CONFIG_H */
