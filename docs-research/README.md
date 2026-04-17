# docs-research/

Research cache for porting flash-moe to Qwen3.6-35B-A3B. Not upstream docs.

Parent plan: [/Users/retry/.claude/plans/groovy-wiggling-waterfall.md](/Users/retry/.claude/plans/groovy-wiggling-waterfall.md)

---

## Diffs

| File | What it is |
|------|-----------|
| [pr-3-runtime-config.diff](pr-3-runtime-config.diff) | PR #3 — ModelConfig struct replaces ~54 hardcoded #defines; adds --model flag and model_manager.py (4,838-line diff, open unmerged) |
| [pr-11-pure-wins.diff](pr-11-pure-wins.diff) | PR #11 — pure perf wins: partial softmax routing, fused kernels, no architectural changes |
| [pr-13-qwen3-coder-next.diff](pr-13-qwen3-coder-next.diff) | PR #13 — Qwen3-Coder-Next port + dequant_matvec_8bit Metal kernel (closed unmerged; kernel subset lives in PR #14) |
| [pr-14-8bit-dequant.diff](pr-14-8bit-dequant.diff) | PR #14 — dequant_matvec_8bit kernel alone; open, the piece we cherry-pick for 8-bit weight support |

## Research notes

| File | What it is |
|------|-----------|
| [fork-nerds-odd-e.md](fork-nerds-odd-e.md) | Summary of nerds-odd-e/flash-moe fork — most active fork, de-hardcodes model ID, already runs Qwen3-Coder-480B |
| [fork-gorroai.md](fork-gorroai.md) | Summary of gorroai/flash-moe fork — benchmarking focus, K=4 routing fix (+5.6%), M2R2 AoT experiment |
| [issue-15-setup-gotchas.md](issue-15-setup-gotchas.md) | GitHub issue #15 verbatim — full setup guide with disk budget math and every known gotcha |
| [issue-17-expert-index.md](issue-17-expert-index.md) | GitHub issue #17 verbatim — incomplete expert_index.json (layer 0 only); must regenerate all layers |
| [issue-20-other-qwen-models.md](issue-20-other-qwen-models.md) | GitHub issue #20 verbatim + our answer — yes, same architecture family works for Qwen3.6 and Qwen3-Coder variants |
| [qwen3.6-35b-a3b-arch.md](qwen3.6-35b-a3b-arch.md) | Full architecture spec for Qwen3.6-35B-A3B — config params, tensor naming, layer types, tokenizer, quant details |
| [flash-moe-hardcoded-constants.md](flash-moe-hardcoded-constants.md) | Map of every hardcoded constant in upstream flash-moe with file:line citations — the porting checklist |
| [port-plan.md](port-plan.md) | Condensed 5-step port plan with disk budget and expected performance figures |
