# Pipeline Guard Audit — infer.m (qwen36-35b-a3b WIP)

Cross-profile audit (MODEL_QWEN35_397B, MODEL_QWEN3_CODER_480B, MODEL_QWEN36_35B_A3B)
of every Metal pipeline/buffer creation site vs every dispatch site, hunting for
the `sigmoid_gate_pipe`-class mismatch (creation guard weaker than dispatch →
`setComputePipelineState:nil` SIGSEGV).

## Verdict

**No additional crash-class macro mismatches found.** The prior fix (`sigmoid_gate_pipe`
+ `buf_attn_gate` now guarded by `HAS_ATTN_GATE || HAS_ATTN_OUTPUT_GATE` at lines
944, 1042, 1180; dispatch at 5131) is complete and consistent.

All three profiles compile cleanly (`clang -c -O0 -fobjc-arc ...`) with only
pre-existing `-Wunused-function` warnings.

## Pipeline / buffer table

| Pipeline / Buffer          | Creation guard                    | Dispatch guard(s)                                            | Verdict      |
|----------------------------|-----------------------------------|--------------------------------------------------------------|--------------|
| matvec_v3                  | unconditional (1029)              | `#if BITS!=8` (1310, 1398, 1456, 1522); runtime `expert_pipe` | OK           |
| matvec_v5                  | unconditional (1030)              | never dispatched                                             | DEAD-CODE    |
| matvec_fast                | unconditional (1031)              | `#if BITS!=8` + runtime `in_dim>4096`; 5176                  | OK           |
| matvec_2bit                | unconditional (1032)              | runtime `g_use_2bit` (1556/1652/1752/1973)                   | OK           |
| matvec_8bit                | unconditional (1033)              | `#if BITS==8` (1294/1513/5163); runtime `s->bits==8` (1395/1453) | OK       |
| rms_norm_sum               | unconditional (1034)              | 5210, 5705                                                   | OK           |
| rms_norm_apply             | unconditional (1035)              | never dispatched                                             | DEAD-CODE    |
| rms_norm_apply_bf16        | unconditional (1036)              | 5226, 5722                                                   | OK           |
| residual_add               | unconditional (1037)              | 5195                                                         | OK           |
| swiglu                     | unconditional (1038)              | 1601/1697/1801/1893/2034/5620 (some inside HAS_SHARED_EXPERT) | OK          |
| attn_scores/softmax/values | unconditional (1039–1041)         | 5086/5105/5116 (runtime fa_idx check)                        | OK           |
| sigmoid_gate_pipe          | `HAS_ATTN_GATE \|\| HAS_ATTN_OUTPUT_GATE` (1042) | same (5131)                                | OK (post-fix)|
| buf_attn_gate              | `HAS_ATTN_GATE \|\| HAS_ATTN_OUTPUT_GATE` (1180) | same (4746, 5138)                           | OK           |
| moe_combine_residual       | unconditional (1045)              | runtime null-check (5653), dispatch 5682                     | OK           |
| delta_net_step             | `HAS_LINEAR_ATTENTION` (1047)     | same (4323, 4474, 4886)                                      | OK           |
| conv1d_step                | `HAS_LINEAR_ATTENTION` (1048)     | same (4275, 4426)                                            | OK           |
| rms_norm_qk                | `HAS_LINEAR_ATTENTION` (1049)     | same (4292, 4443)                                            | OK           |
| compute_decay_beta         | `HAS_LINEAR_ATTENTION` (1050)     | same (4307, 4458)                                            | OK           |
| gated_rms_norm             | `HAS_LINEAR_ATTENTION` (1051)     | same (4343, 4494)                                            | OK           |
| buf_shared_{gate,up,act}   | `HAS_SHARED_EXPERT` (1136)        | same (5576, 5621-5623)                                       | OK           |
| buf_shared_out             | unconditional (1145, deliberate)  | unconditional (3992, 5684); zeroed-out via params[8]=-1000 when `!HAS_SHARED_EXPERT` | OK (intentional) |
| buf_delta_{state,conv_state,q,k,v,g_decay,beta,output} | `HAS_LINEAR_ATTENTION` (1191+) | same | OK                              |
| buf_conv_input             | `HAS_LINEAR_ATTENTION` (1207)     | never referenced                                             | DEAD-CODE    |
| buf_kv_k / buf_kv_v        | unconditional (1169); `[NUM_FULL_ATTN_LAYERS]` | 5088/5118; runtime fa_idx check             | OK           |
| buf_multi_expert_*         | unconditional (1117–1132)         | indexed 0..MAX_K-1 + valid[] runtime check                   | OK           |
| buf_expert_* (legacy)      | unconditional (1091–1101)         | `gpu_expert_forward` legacy path                             | OK           |

## Patches applied

None. Audit is read-only; no macro-mismatch crashes found to patch.

The baseline commit (`ee1cf02`) syncs this otherwise-out-of-date worktree branch to
the qwen36-35b-a3b WIP source so the audit runs against the right code. No changes
beyond the sync.

## Counterfactual: if sigmoid_gate fix had not been applied

The crash window requires all of:
- Profile: `MODEL_QWEN36_35B_A3B` (only profile with `HAS_ATTN_OUTPUT_GATE=1 && HAS_ATTN_GATE=0`).
- seq_len ≥ 32: enables `gpu_attn_ready` (line 4740). Shorter contexts use CPU attention path (lines 2383-2389, guarded by `HAS_ATTN_OUTPUT_GATE` — correct).
- Full-attention layer (indices 3,7,11,...,39). Linear layers skip this path.
- K value is irrelevant.

Matches the reported symptom exactly.

## Similar bug classes — methodology

1. **Asymmetric guards**: after adding any `HAS_X`, grep `#if HAS_X` and confirm creation count ≡ dispatch count for every resource it gates. Preprocess each profile with `clang -E -P -D<PROFILE> ... | grep -E "makePipe|setComputePipelineState" | sort -u` and diff across profiles — asymmetries indicate miss or dead-code.

2. **"Weakest guard wins"**: one resource serving N features must use `HAS_A || HAS_B` on creation AND dispatch AND any derived `#define` (e.g., `Q_PROJ_DIM` in model_config.h:400 correctly uses the disjunction).

3. **BatchMatvecSpec literals** (secondary, not a pipeline guard issue): lines 2803, 5250, 5317 omit the `bits` field in `!HAS_SHARED_EXPERT` branches, zero-init gives `bits=0` → 4-bit dispatch. Clean under default `-Wall` but fails `-Werror=missing-field-initializers`. Only triggers on Coder profile. Recommend adding that flag to Makefile.

4. **Shader shared-memory ceiling**: `threadgroup float x_shared[4096]` in matvec_v3/fast silently OOB-writes for `in_dim > 4096`. Coder has HIDDEN_DIM=6144 — dispatcher routes >4096 to matvec_fast, but matvec_fast has the same ceiling. Future models with HIDDEN_DIM>4096 need a new shader variant.

5. **Runtime null checks as defense-in-depth, not substitution**: paths like lines 4722/4875/5653 runtime-null-check pipelines. Keep both layers; removing the compile guard because "runtime covers it" breaks when pipeline compilation fails for unrelated reasons.
