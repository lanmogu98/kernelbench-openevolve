# OpenEvolve with MLX Metal Kernel Optimization Verification Plan - Gemini

## 1. Overview

This verification process aims to confirm that:

1. **OpenEvolve** is correctly installed and integrated with **MLX** on your MacBook M2 Air.
2. The **Qwen-6B Metal Kernel Optimization** example functions as expected.
3. The system correctly interacts with the **Gemini Free Tier API** (`GEMINI_FT_API_KEY`).
4. The entire loop (Kernel generation -> Compilation -> Execution -> Evaluation) works seamlessly.

## 2. Components Involved

### A. Environment
-   **Conda Environment**: `kb-evolve` (Python 3.10+)
-   **Hardware**: Apple M2 Air (Metal GPU)
-   **Frameworks**:
    -   `openevolve` (Evolutionary algorithm framework)
    -   `mlx` (Machine learning framework for Apple Silicon)
    -   `mlx-lm` (Language models on MLX)

### B. Execution Flow
When you run the verification command, the following modules will be invoked in sequence:

1.  **Task Adapter / Runner (`openevolve-run.py`)**:
    -   Loads the configuration from `config.yaml`.
    -   Initializes the `OpenEvolve` engine.
    -   Loads `initial_program.py` as the starting point (baseline kernel).

2.  **LLM Interface (`openevolve/llm`)**:
    -   Connects to Gemini API using `GEMINI_FT_API_KEY`.
    -   **Adaptation**: We will modify the configuration to respect the **15 RPM (Requests Per Minute)** limit of the Free Tier to avoid 429 errors.
    -   Sends the prompt (system message + code context) to Gemini to generate improved Metal kernels.

3.  **Evolution Engine (`openevolve/controller.py`, `openevolve/evolution_trace.py`)**:
    -   Manages the population of kernels.
    -   Applies evolutionary operators (mutation via LLM, crossover).
    -   Tracks the best performing kernels ("Hall of Fame").

4.  **Evaluator (`evaluator.py`)**:
    -   **Compilation**: Compiles the generated Metal code JIT (Just-In-Time) using `mx.fast.metal_kernel`.
    -   **Correctness Check**: Compares the output of the custom kernel against the standard `mlx.nn.FastAttention` or a reference implementation. **Crucial**: If the output doesn't match, the kernel is discarded (score = 0).
    -   **Benchmarking**: Runs the kernel multiple times to measure latency (ms) on the GPU.
    -   **Scoring**: Returns a fitness score based on speedup (latency reduction) compared to the baseline.

## 3. Expected Output

You will see logs indicating the progress of the evolution:

1.  **Initialization**:
    ```text
    [INFO] Starting OpenEvolve...
    [INFO] Loaded initial program (Baseline).
    [INFO] Baseline latency: 12.50 ms
    ```

2.  **Evolution Loop (Iterative)**:
    ```text
    [INFO] Iteration 1/25
    [INFO] Generating 3 new candidates using Gemini...
    [INFO] [LLM] Request sent (tokens: ~4000)...
    [INFO] Candidate 1: Compiled successfully.
    [INFO] Candidate 1: Correctness check PASSED.
    [INFO] Candidate 1: Latency = 11.20 ms (Speedup: 1.12x)
    [INFO] Candidate 2: Compilation FAILED (Error: ...) -> Score: 0
    ...
    [INFO] Best so far: Candidate 1 (1.12x speedup)
    ```

3.  **Completion**:
    -   A summary of the best kernel found.
    -   Generated files in `openevolve_output/`:
        -   `best_program.py`: The optimized Metal kernel code.
        -   `evolution_history.json`: detailed logs of all attempts.

## 4. Verification Steps (Your Instructions)

Once you give the command, I will execute the following:

1.  **Environment Check**:
    -   Verify `mlx` and `openevolve` import successfully in `kb-evolve`.
    -   Check `GEMINI_FT_API_KEY` is set.

2.  **Configuration Adaptation**:
    -   Modify `config.yaml` to lower `parallel_evaluations` and add delays if necessary for the API rate limit.

3.  **Dry Run**:
    -   Run a short version (e.g., 2 iterations) to prove the pipeline works without waiting for full convergence.

4.  **Result Inspection**:
    -   Show you the `best_program.py` and the speedup achieved.

---
**Ready for your instruction to proceed.**

