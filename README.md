# KernelBench × OpenEvolve Integration

Research project exploring evolutionary GPU kernel optimization by integrating [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluation methodology with [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve).

**Core question**: Can LLM-driven evolutionary search (OpenEvolve) produce GPU kernels that outperform hand-tuned baselines, when guided by rigorous benchmarking (KernelBench)?

---

## Progress

### Phase 1: Evaluation Validity Fixes (Completed)

**PR [#377](https://github.com/algorithmicsuperintelligence/openevolve/pull/377)** — merged into upstream, fixing [issue #372](https://github.com/algorithmicsuperintelligence/openevolve/issues/372).

The existing `mlx_metal_kernel_opt` example had critical evaluation bugs that invalidated all prior results:

| Fix | Problem | Impact |
|-----|---------|--------|
| Subprocess kernel hook | Evolved kernels were silently ignored in benchmark subprocesses | Benchmarks were measuring baseline, not evolved code |
| bfloat16 correctness gate | Correctness tests used float32 inputs | Kernels passed correctness but failed at actual inference dtype |
| Architecture alignment | Docs/prompts assumed 40:8 head ratio | Qwen3-0.6B is actually 16:8 (2:1 GQA) |
| Evaluation flow | No early exit, baseline ran before correctness | Wasted time on broken kernels, GPU state contamination |

**Result after fixes**: 25 evolution iterations, best evolved kernel is **3.2% SLOWER than MLX baseline**. The evolution improved from -11.5% to -3.2% regression but never exceeded baseline, revealing fundamental limitations in the evolution mechanism.

See [`openevolve/examples/mlx_metal_kernel_opt/EVOLUTION_ANALYSIS.md`](openevolve/examples/mlx_metal_kernel_opt/EVOLUTION_ANALYSIS.md) for the full analysis.

### Phase 2: KernelBench-style Evaluation (In Progress)

The evolution analysis identified **feedback quality** as the root cause of failure. The current `combined_score` is meaningless to both the LLM and MAP-Elites selection. KernelBench provides a proven evaluation framework with directly actionable metrics:

| Metric | Current (broken) | KernelBench-style (target) |
|--------|------------------|---------------------------|
| Fitness score | Abstract `combined_score` | Direct `speedup = baseline_time / custom_time` |
| Correctness | Binary pass/fail | Binary + `max_difference`, `avg_difference` |
| Performance | Single number | `mean +/- std` with confidence intervals |
| Population metric | None | `fast_p` (fraction correct AND faster than threshold) |
| Prompt feedback | "Score: 2.96" | "Speedup: 0.85x (15% slower), need > 1.0x" |

Branch: `feature/kernelbench-integration` (early prototype, needs rebase onto current `main`).

### Phase 3: Evolution Mechanism Improvements (Planned)

- **MAP-Elites feature dimensions**: Replace code-length/char-diff with speedup tier, runtime variance, correctness margin
- **GPU profiling integration**: Feed Metal occupancy, bandwidth, cache stats to LLM
- **Maximized LLM context**: Full population + benchmark history instead of 1 parent + 5 samples
- **Domain-specific strategies**: Track tiling, vectorization, memory access patterns
- **Curated bf16 examples**: Reduce the 32% Metal bf16 compilation failure rate

---

## Project Structure

```
kernelbench-openevolve/
├── openevolve/                          # Submodule: fork of algorithmicsuperintelligence/openevolve
│   └── examples/mlx_metal_kernel_opt/   # Main experiment directory
│       ├── evaluator.py                 # Correctness + performance evaluation (fixed)
│       ├── initial_program.py           # Starting Metal kernel
│       ├── config.yaml                  # Evolution config (stable model names)
│       ├── mlx_lm_generate_with_hook.py # Subprocess kernel hook (new)
│       ├── qwen3_benchmark_suite.py     # Benchmark definitions
│       ├── run_evolve_experiment.sh     # Experiment runner (safety fixes)
│       ├── best_program.py              # Committed demo: best evolved kernel
│       ├── best_program_info.json       # Committed demo: metrics snapshot
│       ├── README.md                    # Example guide
│       └── EVOLUTION_ANALYSIS.md        # Detailed failure analysis + future work
├── KernelBench/                         # Submodule: GPU kernel benchmark suite
├── experiments/                         # Local experiment scripts and results
│   ├── generated_kernels/              # LLM-generated CUDA kernels
│   ├── verification_results/           # MLX Metal verification snapshots
│   ├── test_kernel.sh                  # One-click kernel test script
│   └── env_*.sh                        # Environment configs (local/server)
├── mynotes/                             # Personal notes (git-ignored)
│   ├── ACTION_ROADMAP.md               # Original integration roadmap
│   ├── KNOWLEDGE_BACKGROUND_REPORT.md  # Background research
│   ├── OPENEVOLVE_MECHANISM_GUIDE.md   # Deep dive into OpenEvolve internals
│   ├── KERNELBENCH_DEV_REFERENCE.md    # KernelBench dev quick reference
│   ├── OPENSOURCE_CONTRIBUTION_WORKFLOW.md  # Git/PR workflow reference
│   └── ...                             # Issue notes, verification guide, etc.
└── README.md
```

---

## Getting Started

### Clone with Submodules

```bash
git clone --recurse-submodules https://github.com/lanmogu98/kernelbench-openevolve.git
cd kernelbench-openevolve
```

If already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Sync with Upstream

```bash
# openevolve
cd openevolve
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
cd ..

# KernelBench
cd KernelBench
git fetch upstream
git merge upstream/main
cd ..
```

For the full fork + PR workflow, see `mynotes/OPENSOURCE_CONTRIBUTION_WORKFLOW.md` (local only, not tracked in git).

---

## Branch Strategy

| Branch | Status | Purpose |
|--------|--------|---------|
| `main` | Active | Synced with upstream, stable |
| `feature/kernelbench-integration` | Paused | KernelBench evaluation prototype (needs rebase) |

Completed / merged branches are deleted after upstream merge.

---

## Key References

- **Upstream PR**: [#377 — Fix mlx_metal_kernel_opt evaluation validity](https://github.com/algorithmicsuperintelligence/openevolve/pull/377)
- **Upstream Issue**: [#372 — Subprocess benchmarks ignore evolved kernels](https://github.com/algorithmicsuperintelligence/openevolve/issues/372)
- **Evolution Analysis**: [`EVOLUTION_ANALYSIS.md`](openevolve/examples/mlx_metal_kernel_opt/EVOLUTION_ANALYSIS.md)
- **KernelBench**: [github.com/ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- **OpenEvolve**: [github.com/algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)

---

*Last updated: 2026-02-16*
