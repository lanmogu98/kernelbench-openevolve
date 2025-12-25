# -*- coding: utf-8 -*-
"""
gpu_kernel_optimizer_simple.py - LLM-Powered GPU Kernel Optimization Agent (Simple Version)

==============================================================================
VERSION HISTORY
==============================================================================

v8.0.0 - ROBUST EXTRACTION: Enhanced JSON/code extraction with auto-retry
       - Multi-pattern JSON extraction (handles markdown-wrapped, prefixed, etc.)
       - Fallback code extraction from various markdown formats
       - Auto-retry mechanism: if extraction fails, send simplified follow-up
       - Up to 3 retries with progressively simpler prompts
       - All v7 features preserved (full history, 20+ models)

v7.0.0 - FULL HISTORY: LLM sees ALL historical versions' complete data
v6.0.0 - Extended model support (20+ top-tier models)
v5.0.0 - History tracking with intent/analysis
v2.0.0 - Multi-task support
v1.0.0 - Initial version

==============================================================================
KEY IMPROVEMENTS (v8)
==============================================================================

1. MULTI-PATTERN JSON EXTRACTION:
   - Standard JSON: {...}
   - Markdown-wrapped: ```json {...} ```
   - With preamble: "Here is my response: {...}"
   - Nested in text: extracts deepest valid JSON

2. ROBUST CODE EXTRACTION:
   - From JSON "code" field
   - From ```python ... ``` blocks
   - From ```triton ... ``` blocks
   - From ``` ... ``` blocks (generic)
   - Raw code detection (import statements)

3. AUTO-RETRY MECHANISM:
   - If extraction fails, send simplified follow-up prompt
   - "Please provide ONLY the Python code, no explanation"
   - Up to 3 retries with increasingly direct prompts
   - Prevents "Retry after generation failure" issues

==============================================================================
USAGE
==============================================================================

export OPENROUTER_API_KEY=your_key

python3 gpu_kernel_optimizer_simple.py --task flashattention --model gemini3-pro --suffix v1
python3 gpu_kernel_optimizer_simple.py --task flashattention --all-models --suffix v1
python3 gpu_kernel_optimizer_simple.py --list-models

==============================================================================
"""

import os
import re
import json
import time
import logging
import subprocess
import csv
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai


# --------------------------------------------------------------------------- #
# 1. TASK CONFIGURATIONS
# --------------------------------------------------------------------------- #

TASK_CONFIGS = {
    "histogram": {
        "leaderboard": "histogram",
        "description": """
Multi-channel Histogram: For each channel, produce histogram of values.
Input: array [1048576, 512], num_bins=256
Output: histogram [512, 256]
Test: length=1048576, channels=512, bins=256, seed=1001
""",
        "interface": """
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data  # array: [length, channels]
    return histogram  # [channels, bins]
"""
    },
    "flashattention": {
        "leaderboard": "flashattention",
        "description": """
Flash Attention: Efficient attention with tiling and online softmax.
Attention(Q,K,V) = softmax(QK^T/sqrt(d)) V
Input: Q,K,V shape (batch, heads, seq, dim)
Tests: (1,64,1024,128), (2,64,4096,128), (4,64,8192,128)
""",
        "interface": """
from task import input_t, output_t
import torch, triton, triton.language as tl

@triton.jit
def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, ...):
    pass

def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    return output
"""
    }
}


# --------------------------------------------------------------------------- #
# 2. MODEL CONFIGURATIONS (20+ models)
# --------------------------------------------------------------------------- #

MODEL_CONFIGS = {
    # TIER 1: Flagship
    "gpt5": {"name": "openai/gpt-5.1", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gpt5", "tier": "tier1", "description": "OpenAI GPT-5.1"},
    "claude-opus": {"name": "anthropic/claude-4.5-opus", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "claude-opus", "tier": "tier1", "description": "Claude 4.5 Opus"},
    "gemini3-pro": {"name": "google/gemini-3-pro-preview-20251117", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gemini3-pro", "tier": "tier1", "description": "Gemini 3.0 Pro"},
    "deepseek-r1": {"name": "deepseek/deepseek-r1-0528", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "deepseek-r1", "tier": "tier1", "description": "DeepSeek R1"},
    
    # TIER 2: Excellent
    "claude-sonnet": {"name": "anthropic/claude-4.5-sonnet-20250929", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "claude-sonnet", "tier": "tier2", "description": "Claude 4.5 Sonnet"},
    "gpt4.1": {"name": "openai/gpt-4.1", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gpt4.1", "tier": "tier2", "description": "GPT-4.1"},
    "gemini25-pro": {"name": "google/gemini-2.5-pro", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gemini25-pro", "tier": "tier2", "description": "Gemini 2.5 Pro"},
    "qwen3-235b": {"name": "qwen/qwen3-235b-a22b-instruct", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "qwen3-235b", "tier": "tier2", "description": "Qwen3-235B"},
    "grok4": {"name": "x-ai/grok-4", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "grok4", "tier": "tier2", "description": "Grok-4"},
    "glm46": {"name": "z-ai/glm-4.6", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "glm46", "tier": "tier2", "description": "GLM-4.6"},
    "claude4-sonnet": {"name": "anthropic/claude-4-sonnet-20250522", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "claude4-sonnet", "tier": "tier2", "description": "Claude 4 Sonnet"},
    
    # TIER 3: Good
    "qwq-32b": {"name": "qwen/qwq-32b", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "qwq-32b", "tier": "tier3", "description": "QwQ-32B"},
    "gemini25-flash": {"name": "google/gemini-2.5-flash", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gemini25-flash", "tier": "tier3", "description": "Gemini 2.5 Flash"},
    "llama4-maverick": {"name": "meta-llama/llama-4-maverick", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "llama4-maverick", "tier": "tier3", "description": "Llama 4 Maverick"},
    "mistral-large": {"name": "mistralai/mistral-large-latest", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "mistral-large", "tier": "tier3", "description": "Mistral Large"},
    "deepseek-v3": {"name": "deepseek/deepseek-chat-v3-0324", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "deepseek-v3", "tier": "tier3", "description": "DeepSeek V3"},
    "gpt4o": {"name": "openai/gpt-4o", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gpt4o", "tier": "tier3", "description": "GPT-4o"},
    "grok3": {"name": "x-ai/grok-3", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "grok3", "tier": "tier3", "description": "Grok-3"},
    "qwen3-32b": {"name": "qwen/qwen3-32b-instruct", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "qwen3-32b", "tier": "tier3", "description": "Qwen3-32B"},
    "gpt5-mini": {"name": "openai/gpt-5-mini-2025-08-07", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "gpt5-mini", "tier": "tier3", "description": "GPT-5 Mini"},
    
    # Aliases
    "claude": {"name": "anthropic/claude-4.5-opus", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "claude-opus", "tier": "tier1", "description": "Alias for claude-opus"},
    "deepseek": {"name": "deepseek/deepseek-r1-0528", "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "short_name": "deepseek-r1", "tier": "tier1", "description": "Alias for deepseek-r1"},
}

MODEL_TIERS = {
    "tier1": ["gpt5", "claude-opus", "gemini3-pro", "deepseek-r1"],
    "tier2": ["claude-sonnet", "gpt4.1", "gemini25-pro", "qwen3-235b", "grok4", "glm46", "claude4-sonnet"],
    "tier3": ["qwq-32b", "gemini25-flash", "llama4-maverick", "mistral-large", "deepseek-v3", "gpt4o", "grok3", "qwen3-32b", "gpt5-mini"],
}
ALL_MODELS = MODEL_TIERS["tier1"] + MODEL_TIERS["tier2"] + MODEL_TIERS["tier3"]


# --------------------------------------------------------------------------- #
# 3. DATA STRUCTURES
# --------------------------------------------------------------------------- #

@dataclass
class OptimizationConfig:
    task: str = "histogram"
    leaderboard: str = "histogram"
    task_description: str = ""
    task_interface: str = ""
    model_name: str = "openai/gpt-5.1"
    model_short: str = "gpt5"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    optimizer_type: str = "simple"
    max_iterations: int = 30
    reflection_interval: int = 5
    use_profile: bool = True
    output_dir: str = "./optimization_output"
    submission_prefix: str = "submission_auto_v"


@dataclass
class SubmissionResult:
    version: int
    timestamp: str
    mode: str
    success: bool = False
    passed_tests: bool = False
    timing_ms: Optional[float] = None
    timing_std: Optional[float] = None
    ranked_timing_ms: Optional[float] = None
    error_message: str = ""
    raw_output: str = ""
    profile_data: Optional[Dict] = None


@dataclass 
class CodeVersion:
    """Stores COMPLETE information for each version - nothing truncated."""
    version: int
    code: str  # FULL code, never truncated
    timestamp: str
    intended_change: str = ""
    reasoning: str = ""
    test_result: Optional[SubmissionResult] = None
    leaderboard_result: Optional[SubmissionResult] = None
    profile_result: Optional[SubmissionResult] = None
    timing_change: Optional[float] = None
    profile_interpretation: str = ""
    result_analysis: str = ""


class OptimizationMemory:
    def __init__(self):
        self.insights: List[str] = []
        self.best_timing_ms: float = float('inf')
        self.best_version: int = 0
        self.best_code: str = ""
        self.failed_approaches: List[str] = []
        self.successful_patterns: List[str] = []
    
    def add_insight(self, insight: str):
        if insight not in self.insights:
            self.insights.append(insight)
            if len(self.insights) > 30:
                self.insights = self.insights[-30:]
    
    def to_dict(self) -> Dict:
        return {
            "insights": self.insights[-15:],
            "best_timing_ms": self.best_timing_ms,
            "best_version": self.best_version,
            "failed_approaches": self.failed_approaches[-15:],
            "successful_patterns": self.successful_patterns[-15:]
        }


# --------------------------------------------------------------------------- #
# 4. ROBUST EXTRACTION (v8 NEW)
# --------------------------------------------------------------------------- #

class RobustExtractor:
    """
    v8 NEW: Multi-pattern extraction with fallbacks.
    Handles various LLM output formats robustly.
    """
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        """
        Try multiple patterns to extract JSON from LLM response.
        Returns parsed dict or None if all patterns fail.
        """
        if not text:
            return None
        
        # Pattern 1: JSON in markdown code block
        patterns = [
            r'```json\s*([\s\S]*?)```',           # ```json ... ```
            r'```\s*({\s*[\s\S]*?})\s*```',       # ``` {...} ```
            r'`({[\s\S]*?})`',                     # `{...}`
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        
        # Pattern 2: Find all JSON-like structures and try each
        # Start from the largest to get complete JSON
        json_candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        
        # Sort by length (descending) to try larger JSONs first
        json_candidates.sort(key=len, reverse=True)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                # Verify it has expected fields
                if isinstance(parsed, dict) and ('code' in parsed or 'intended_change' in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Pattern 3: Greedy match from first { to last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def extract_code(text: str, json_data: Optional[Dict] = None) -> str:
        """
        Extract Python code from response using multiple strategies.
        """
        # Strategy 1: From JSON "code" field
        if json_data and 'code' in json_data:
            code = json_data['code']
            if code and len(code.strip()) > 50:  # Sanity check
                return code.strip()
        
        # Strategy 2: Python code block
        patterns = [
            r'```python\s*([\s\S]*?)```',         # ```python ... ```
            r'```triton\s*([\s\S]*?)```',         # ```triton ... ```
            r'```py\s*([\s\S]*?)```',             # ```py ... ```
            r'```\s*(import[\s\S]*?)```',         # ``` import... ``` (code starting with import)
            r'```\s*(@triton[\s\S]*?)```',        # ``` @triton... ```
            r'```\s*(from task[\s\S]*?)```',      # ``` from task... ```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                if len(code) > 50:  # Sanity check
                    return code
        
        # Strategy 3: Generic code block
        match = re.search(r'```\s*([\s\S]*?)```', text)
        if match:
            code = match.group(1).strip()
            # Check if it looks like Python code
            if ('import' in code or 'def ' in code or '@triton' in code) and len(code) > 50:
                return code
        
        # Strategy 4: Raw code detection (no markdown)
        # Look for code that starts with common patterns
        raw_patterns = [
            r'(import torch[\s\S]+def custom_kernel[\s\S]+)',
            r'(from task import[\s\S]+def custom_kernel[\s\S]+)',
            r'(@triton\.jit[\s\S]+def custom_kernel[\s\S]+)',
        ]
        
        for pattern in raw_patterns:
            match = re.search(pattern, text)
            if match:
                code = match.group(1).strip()
                if len(code) > 100:
                    return code
        
        return ""
    
    @staticmethod
    def extract_field(text: str, field: str, json_data: Optional[Dict] = None) -> str:
        """
        Extract a specific field from response.
        """
        # From JSON
        if json_data and field in json_data:
            return str(json_data[field])
        
        # Pattern matching for specific fields
        patterns = [
            rf'"{field}"\s*:\s*"([^"]*)"',        # "field": "value"
            rf"'{field}'\s*:\s*'([^']*)'",        # 'field': 'value'
            rf'{field}:\s*([^\n]+)',               # field: value (YAML-like)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""


# --------------------------------------------------------------------------- #
# 5. POPCORN CLI
# --------------------------------------------------------------------------- #

class PopcornCLI:
    def __init__(self, leaderboard: str, timeout: int = 300):
        self.leaderboard = leaderboard
        self.timeout = timeout
    
    def submit(self, filepath: str, mode: str = "test") -> SubmissionResult:
        cmd = ["popcorn-cli", "submit", "--leaderboard", self.leaderboard, "--mode", mode, filepath]
        version = int(re.search(r'v(\d+)', filepath).group(1)) if re.search(r'v(\d+)', filepath) else 0
        timestamp = datetime.now().isoformat()
        
        for attempt in range(3):
            try:
                logging.info(f"Submitting {filepath} in {mode} mode...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                output = result.stdout + result.stderr
                
                if "503" in output or "service unavailable" in output.lower():
                    if attempt < 2:
                        time.sleep(30 * (2 ** attempt))
                        continue
                    return SubmissionResult(version=version, timestamp=timestamp, mode=mode,
                                          success=False, error_message="503", raw_output=output)
                return self._parse(output, version, timestamp, mode)
            except subprocess.TimeoutExpired:
                if attempt < 2: time.sleep(30); continue
                return SubmissionResult(version=version, timestamp=timestamp, mode=mode, success=False, error_message="Timeout")
            except Exception as e:
                return SubmissionResult(version=version, timestamp=timestamp, mode=mode, success=False, error_message=str(e))
        return SubmissionResult(version=version, timestamp=timestamp, mode=mode, success=False)
    
    def _parse(self, output: str, version: int, timestamp: str, mode: str) -> SubmissionResult:
        r = SubmissionResult(version=version, timestamp=timestamp, mode=mode, success="✅" in output, raw_output=output)
        r.passed_tests = "✅ Passed" in output or "✅ Testing successful" in output
        
        if "❌" in output:
            r.passed_tests = False
            for pat in [r'## Test log:\s*```([\s\S]*?)```', r'(Traceback[\s\S]*?Error:.*?)(?=\n\n|\Z)', r'❌[\s\S]*?(?=\n\n\n|\Z)']:
                if m := re.search(pat, output):
                    r.error_message = m.group(1 if '```' in pat else 0).strip()[:2000]
                    break
        
        if m := re.search(r'⏱\s*(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*ms', output):
            r.timing_ms, r.timing_std = float(m.group(1)), float(m.group(2))
        if m := re.search(r'Ranked.*?⏱\s*(\d+\.?\d*)', output, re.DOTALL):
            r.ranked_timing_ms = float(m.group(1))
        if mode == "profile":
            r.profile_data = self._parse_profile(output)
        return r
    
    def _parse_profile(self, output: str) -> Dict:
        profile = {"kernels": [], "raw": output[:5000]}
        for section in re.split(r'Kernel \d+:', output)[1:]:
            k = {}
            if m := re.search(r'^\s*(\w+)', section): k["name"] = m.group(1)
            for name, pat in [("SM_Busy", r'SM.*?(\d+\.?\d*)%'), ("DRAM", r'DRAM.*?(\d+\.?\d*)%'),
                             ("L1", r'L1.*?(\d+\.?\d*)%'), ("L2", r'L2.*?(\d+\.?\d*)%'),
                             ("Occupancy", r'[Oo]ccupancy.*?(\d+\.?\d*)%')]:
                if m := re.search(pat, section, re.I): k[name] = float(m.group(1))
            if k: profile["kernels"].append(k)
        return profile


# --------------------------------------------------------------------------- #
# 6. LLM OPTIMIZER - v8 WITH RETRY
# --------------------------------------------------------------------------- #

class KernelOptimizer:
    """
    v8: Full history + robust extraction + auto-retry mechanism.
    """
    
    # Retry prompts (progressively simpler)
    RETRY_PROMPTS = [
        "Your previous response could not be parsed. Please respond with ONLY a JSON object containing 'intended_change' and 'code' fields. No other text.",
        "Please provide ONLY the complete Python code for the Triton kernel. No explanations, no JSON, just the code starting with 'import'.",
        "Output the Python code only. Start with 'import torch' or 'from task import'.",
    ]
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=os.getenv(config.api_key_env), base_url=config.api_base)
        self._last_reasoning = ""
        self._last_response = ""
        self.h100 = "H100: 132 SMs, 80GB HBM3 @ 3.35 TB/s, 256KB SMEM/SM, warp=32"
        self.extractor = RobustExtractor()
    
    def generate_initial_code(self) -> Tuple[str, str, str]:
        prompt = f"""You are an expert GPU kernel programmer. Generate an initial Triton kernel implementation.

TASK:
{self.config.task_description}

TARGET HARDWARE: {self.h100}

REQUIRED INTERFACE:
```python
{self.config.task_interface}
```

REQUIREMENTS:
1. Use Triton (@triton.jit, triton.language as tl)
2. Focus on correctness first, then optimize
3. Use appropriate block sizes and memory access patterns

Respond in JSON format:
{{
    "intended_change": "Describe your implementation strategy",
    "code": "Complete Python code with Triton kernel"
}}"""
        
        code, intended, reasoning = self._call_llm_with_retry(prompt)
        return code, reasoning, intended or "Initial implementation"
    
    def optimize_code(self, current_code: str, history: List[CodeVersion], mem: OptimizationMemory,
                      latest_result: Optional[SubmissionResult], profile_data: Optional[Dict]) -> Tuple[str, str, str]:
        """
        v8: Passes COMPLETE history to LLM with retry on extraction failure.
        """
        
        full_history = self._build_full_history(history)
        current_status = self._build_current_status(latest_result, profile_data)
        
        prompt = f"""You are an expert GPU kernel optimizer. Improve performance based on the COMPLETE history below.

TASK:
{self.config.task_description}

TARGET HARDWARE: {self.h100}

REQUIRED INTERFACE:
```python
{self.config.task_interface}
```

================================================================================
COMPLETE OPTIMIZATION HISTORY
================================================================================
BEST RESULT SO FAR: {mem.best_timing_ms:.2f}ms (Version {mem.best_version})

{full_history}
================================================================================

CURRENT STATUS:
{current_status}

INSIGHTS:
{json.dumps(mem.to_dict(), indent=2)}

YOUR TASK:
1. Study the history - identify what worked and what didn't
2. If tests failed: Fix the error
3. If tests passed: Optimize based on profile data
4. The best implementation might be from an earlier version - review it

Respond in JSON:
{{
    "intended_change": "What specific change I will make and why",
    "code": "Complete Python code"
}}"""
        
        code, intended, reasoning = self._call_llm_with_retry(prompt)
        return code, reasoning, intended or reasoning[:200]
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Tuple[str, str, str]:
        """
        v8 NEW: Call LLM with automatic retry on extraction failure.
        Returns (code, intended_change, reasoning).
        """
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries + 1):
            try:
                # Make API call
                content, reasoning = self._call_llm_raw(messages)
                
                if not content:
                    logging.warning(f"  [Attempt {attempt + 1}] Empty response")
                    if attempt < max_retries:
                        messages.append({"role": "assistant", "content": "(empty response)"})
                        messages.append({"role": "user", "content": self.RETRY_PROMPTS[min(attempt, len(self.RETRY_PROMPTS) - 1)]})
                        continue
                    return "", "", "Empty response after retries"
                
                # Try to extract JSON
                json_data = self.extractor.extract_json(content)
                
                # Try to extract code
                code = self.extractor.extract_code(content, json_data)
                
                if code:
                    intended = ""
                    if json_data:
                        intended = json_data.get("intended_change", "") or json_data.get("analysis", "")
                    if not intended:
                        intended = self.extractor.extract_field(content, "intended_change", json_data)
                    
                    logging.info(f"  [Attempt {attempt + 1}] Extraction successful (code: {len(code)} chars)")
                    self._last_response = content
                    self._last_reasoning = reasoning
                    return code, intended, reasoning
                
                # Extraction failed - retry
                logging.warning(f"  [Attempt {attempt + 1}] Code extraction failed, retrying...")
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": self.RETRY_PROMPTS[min(attempt, len(self.RETRY_PROMPTS) - 1)]})
                
            except Exception as e:
                logging.error(f"  [Attempt {attempt + 1}] Error: {e}")
                if attempt >= max_retries:
                    return "", "", str(e)
                time.sleep(2)
        
        logging.error(f"  All {max_retries + 1} attempts failed")
        return "", "", "Extraction failed after all retries"
    
    def _call_llm_raw(self, messages: List[Dict]) -> Tuple[str, str]:
        """
        Raw LLM API call without retry logic.
        Returns (content, reasoning).
        """
        model_lower = self.config.model_name.lower()
        is_reasoning = any(x in model_lower for x in ["r1", "qwq", "reasoner"])
        is_gpt5 = "gpt-5" in model_lower
        is_openrouter = "openrouter" in self.config.api_base.lower()
        
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": 16000
        }
        
        if is_reasoning:
            params["max_tokens"] = 32000
        elif is_gpt5 and is_openrouter:
            params["extra_body"] = {"reasoning": {"enabled": True}}
            params["temperature"] = 0.7
        else:
            params["temperature"] = 0.7
        
        resp = self.client.chat.completions.create(**params)
        content = resp.choices[0].message.content or ""
        
        # Extract reasoning if available
        msg = resp.choices[0].message
        reasoning = ""
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            reasoning = msg.reasoning_content
        elif hasattr(msg, 'reasoning_details') and msg.reasoning_details:
            reasoning = "\n".join(
                d.get('content', str(d)) if isinstance(d, dict) else str(d)
                for d in msg.reasoning_details
            )
        
        if reasoning:
            logging.info(f"    [Thinking]: {len(reasoning)} chars")
        logging.info(f"    [Response]: {len(content)} chars")
        
        return content, reasoning
    
    def _build_full_history(self, history: List[CodeVersion]) -> str:
        """Build COMPLETE history with ALL versions' FULL data."""
        if not history:
            return "No history yet - this is the first version."
        
        sections = []
        
        for v in history:
            status = "PASS" if v.test_result and v.test_result.passed_tests else "FAIL"
            timing_str = f"{v.leaderboard_result.timing_ms:.2f}ms" if v.leaderboard_result and v.leaderboard_result.timing_ms else "N/A"
            delta_str = f" (Δ{v.timing_change:+.2f}ms)" if v.timing_change is not None else ""
            
            section = f"""
{'='*60}
VERSION {v.version} [{status}] {timing_str}{delta_str}
{'='*60}

INTENDED CHANGE:
{v.intended_change}

RESULT ANALYSIS:
{v.result_analysis}

PROFILE INTERPRETATION:
{v.profile_interpretation or 'N/A'}
"""
            
            if v.profile_result and v.profile_result.profile_data:
                profile_json = json.dumps(v.profile_result.profile_data, indent=2)
                section += f"""
FULL PROFILE DATA:
{profile_json}
"""
            
            if v.test_result and not v.test_result.passed_tests and v.test_result.error_message:
                section += f"""
ERROR MESSAGE:
{v.test_result.error_message}
"""
            
            section += f"""
COMPLETE CODE:
```python
{v.code}
```
"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _build_current_status(self, result: Optional[SubmissionResult], profile: Optional[Dict]) -> str:
        parts = []
        if result:
            parts.append(f"Passed: {result.passed_tests}")
            parts.append(f"Timing: {result.timing_ms}ms ± {result.timing_std}ms" if result.timing_ms else "Timing: N/A")
            if result.error_message:
                parts.append(f"Error: {result.error_message}")
        if profile:
            parts.append(f"Profile: {json.dumps(profile, indent=2)}")
        return "\n".join(parts) if parts else "No current status"
    
    def analyze_result(self, v: CodeVersion, prev: Optional[CodeVersion]) -> Tuple[str, str]:
        """Generate profile interpretation and result analysis."""
        interp = ""
        analysis = ""
        
        if v.profile_result and v.profile_result.profile_data:
            parts = []
            for k in v.profile_result.profile_data.get("kernels", []):
                name = k.get("name", "kernel")
                sm = k.get("SM_Busy", 0)
                dram = k.get("DRAM", 0)
                occ = k.get("Occupancy", 0)
                
                if sm < 30:
                    parts.append(f"{name}: SM={sm:.0f}% (VERY LOW - latency bound)")
                elif sm < 50:
                    parts.append(f"{name}: SM={sm:.0f}% (LOW)")
                elif sm < 70:
                    parts.append(f"{name}: SM={sm:.0f}% (MODERATE)")
                else:
                    parts.append(f"{name}: SM={sm:.0f}% (GOOD)")
                
                if dram > 80:
                    parts.append(f"  DRAM={dram:.0f}% (MEMORY BOUND)")
                elif dram > 50:
                    parts.append(f"  DRAM={dram:.0f}% (moderate)")
                elif dram > 0:
                    parts.append(f"  DRAM={dram:.0f}%")
                
                if occ > 0:
                    parts.append(f"  Occupancy={occ:.0f}%")
            
            interp = "; ".join(parts)
        
        if v.test_result and not v.test_result.passed_tests:
            analysis = f"FAILED: {v.test_result.error_message[:200]}"
        elif v.leaderboard_result and v.leaderboard_result.timing_ms:
            t = v.leaderboard_result.timing_ms
            if prev and prev.leaderboard_result and prev.leaderboard_result.timing_ms:
                pt = prev.leaderboard_result.timing_ms
                d = t - pt
                pct = (d / pt) * 100 if pt > 0 else 0
                if d < -0.5:
                    analysis = f"IMPROVED: {t:.2f}ms (was {pt:.2f}ms, {d:.2f}ms / {pct:.1f}% faster)"
                elif d > 0.5:
                    analysis = f"REGRESSED: {t:.2f}ms (was {pt:.2f}ms, +{d:.2f}ms / {pct:.1f}% slower)"
                else:
                    analysis = f"NO CHANGE: {t:.2f}ms (was {pt:.2f}ms)"
            else:
                analysis = f"BASELINE: {t:.2f}ms (first successful version)"
        
        return interp, analysis
    
    def reflect(self, history: List[CodeVersion], mem: OptimizationMemory) -> List[str]:
        """Periodic reflection on progress."""
        traj = [{
            "version": v.version,
            "passed": v.test_result.passed_tests if v.test_result else None,
            "timing_ms": v.leaderboard_result.timing_ms if v.leaderboard_result else None,
            "intended": v.intended_change[:100],
            "result": v.result_analysis[:100]
        } for v in history[-20:]]
        
        prompt = f"""Analyze this optimization trajectory and extract insights:

{json.dumps(traj, indent=2)}

Current best: {mem.best_timing_ms:.2f}ms (V{mem.best_version})

Respond in JSON:
{{
    "insights": ["insight1", "insight2", ...],
    "failed_approaches": ["what failed and why", ...],
    "successful_patterns": ["what worked and why", ...]
}}"""
        
        try:
            content, _ = self._call_llm_raw([{"role": "user", "content": prompt}])
            json_data = self.extractor.extract_json(content)
            if json_data:
                mem.failed_approaches.extend(json_data.get("failed_approaches", []))
                mem.successful_patterns.extend(json_data.get("successful_patterns", []))
                return json_data.get("insights", [])
        except:
            pass
        return []


# --------------------------------------------------------------------------- #
# 7. LOGGING
# --------------------------------------------------------------------------- #

class OptimizationLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.reasoning_dir = os.path.join(output_dir, "reasoning")
        os.makedirs(self.reasoning_dir, exist_ok=True)
        
        self.csv_file = open(os.path.join(output_dir, "log.csv"), 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["version", "passed", "timing_ms", "timing_change", 
                          "intended_change", "result_analysis", "profile_interpretation", "error"])
    
    def log(self, v: CodeVersion, mem: OptimizationMemory, reasoning: str = "", response: str = ""):
        with open(os.path.join(self.reasoning_dir, f"v{v.version}.txt"), 'w') as f:
            f.write(f"VERSION {v.version}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"INTENDED CHANGE:\n{v.intended_change}\n\n")
            f.write(f"RESULT ANALYSIS:\n{v.result_analysis}\n\n")
            f.write(f"PROFILE INTERPRETATION:\n{v.profile_interpretation}\n\n")
            f.write(f"{'='*60}\n")
            f.write(f"MODEL REASONING:\n{'='*60}\n{reasoning}\n\n")
            f.write(f"{'='*60}\n")
            f.write(f"MODEL RESPONSE:\n{'='*60}\n{response}\n")
        
        timing = v.leaderboard_result.timing_ms if v.leaderboard_result else None
        passed = v.test_result.passed_tests if v.test_result else False
        error = v.test_result.error_message[:200] if v.test_result and v.test_result.error_message else ""
        
        self.csv.writerow([
            v.version, passed, timing, v.timing_change,
            v.intended_change[:300], v.result_analysis[:300],
            v.profile_interpretation[:300], error
        ])
        self.csv_file.flush()
    
    def save_memory(self, mem: OptimizationMemory):
        with open(os.path.join(self.output_dir, "memory.json"), 'w') as f:
            json.dump(mem.to_dict(), f, indent=2)
    
    def close(self):
        self.csv_file.close()


# --------------------------------------------------------------------------- #
# 8. MAIN AGENT
# --------------------------------------------------------------------------- #

class GPUKernelOptimizationAgent:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cli = PopcornCLI(config.leaderboard)
        self.opt = KernelOptimizer(config)
        self.log = OptimizationLogger(config.output_dir)
        self.mem = OptimizationMemory()
        self.history: List[CodeVersion] = []
        self.ver = 0
    
    def run(self):
        logging.info(f"GPU Optimizer v8 (ROBUST EXTRACTION) | {self.config.task} | {self.config.model_name}")
        logging.info(f"Output: {self.config.output_dir}")
        
        # Generate initial code
        code, reasoning, intended = self.opt.generate_initial_code()
        if not code:
            logging.error("Failed to generate initial code after all retries")
            return
        
        self.ver = 1
        
        for i in range(self.config.max_iterations):
            logging.info(f"\n{'='*60}")
            logging.info(f"ITERATION {i+1}/{self.config.max_iterations} (Version {self.ver})")
            logging.info(f"{'='*60}")
            logging.info(f"INTENDED: {intended[:120]}...")
            
            # Save code
            path = os.path.join(self.config.output_dir, f"{self.config.submission_prefix}{self.ver}.py")
            with open(path, 'w') as f:
                f.write(code)
            
            # Create version record
            v = CodeVersion(
                version=self.ver,
                code=code,
                timestamp=datetime.now().isoformat(),
                intended_change=intended,
                reasoning=reasoning
            )
            
            prev = self.history[-1] if self.history else None
            
            # Test
            logging.info("\n--- Testing ---")
            v.test_result = self.cli.submit(path, "test")
            
            if not v.test_result.passed_tests:
                logging.warning(f"FAILED: {v.test_result.error_message[:100]}")
                v.result_analysis = f"FAILED: {v.test_result.error_message[:200]}"
            else:
                logging.info("PASSED ✓")
                
                # Benchmark
                logging.info("\n--- Benchmarking ---")
                v.leaderboard_result = self.cli.submit(path, "leaderboard")
                
                if v.leaderboard_result.timing_ms:
                    logging.info(f"Timing: {v.leaderboard_result.timing_ms:.2f}ms ± {v.leaderboard_result.timing_std:.2f}ms")
                    
                    if prev and prev.leaderboard_result and prev.leaderboard_result.timing_ms:
                        v.timing_change = v.leaderboard_result.timing_ms - prev.leaderboard_result.timing_ms
                        logging.info(f"Delta: {v.timing_change:+.2f}ms")
                    
                    if v.leaderboard_result.timing_ms < self.mem.best_timing_ms:
                        improvement = self.mem.best_timing_ms - v.leaderboard_result.timing_ms
                        self.mem.best_timing_ms = v.leaderboard_result.timing_ms
                        self.mem.best_version = self.ver
                        self.mem.best_code = code
                        logging.info(f"🎉 NEW BEST! (improved by {improvement:.2f}ms)")
                
                # Profile
                if self.config.use_profile:
                    logging.info("\n--- Profiling ---")
                    v.profile_result = self.cli.submit(path, "profile")
                    if v.profile_result.profile_data:
                        kernels = v.profile_result.profile_data.get("kernels", [])
                        logging.info(f"Profiled {len(kernels)} kernel(s)")
                
                # Analyze result
                v.profile_interpretation, v.result_analysis = self.opt.analyze_result(v, prev)
                logging.info(f"RESULT: {v.result_analysis}")
                if v.profile_interpretation:
                    logging.info(f"PROFILE: {v.profile_interpretation[:150]}...")
            
            # Store in history
            self.history.append(v)
            
            # Log
            self.log.log(v, self.mem, self.opt._last_reasoning, self.opt._last_response)
            
            # Periodic reflection
            if (i + 1) % self.config.reflection_interval == 0:
                logging.info("\n--- Reflection ---")
                for insight in self.opt.reflect(self.history, self.mem):
                    self.mem.add_insight(insight)
                    logging.info(f"  💡 {insight}")
            
            # Check termination
            if self.mem.best_timing_ms < 1.0:
                logging.info("\n🏆 TARGET ACHIEVED!")
                break
            
            # Generate next version
            logging.info("\n--- Generating Next Version ---")
            profile = v.profile_result.profile_data if v.profile_result else None
            code, reasoning, intended = self.opt.optimize_code(
                code, self.history, self.mem,
                v.leaderboard_result or v.test_result,
                profile
            )
            
            if not code:
                logging.error("Failed to generate code after all retries, stopping")
                break
            
            self.ver += 1
            time.sleep(1)
        
        # Final summary
        self._print_summary()
        self.log.save_memory(self.mem)
        self.log.close()
    
    def _print_summary(self):
        logging.info("\n" + "="*60)
        logging.info("OPTIMIZATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Total versions: {self.ver}")
        logging.info(f"Best timing: {self.mem.best_timing_ms:.2f}ms (Version {self.mem.best_version})")
        
        logging.info("\nProgression:")
        for v in self.history:
            status = "✓" if v.test_result and v.test_result.passed_tests else "✗"
            timing = v.leaderboard_result.timing_ms if v.leaderboard_result else None
            timing_str = f"{timing:.2f}ms" if timing else "N/A"
            delta = f" ({v.timing_change:+.2f})" if v.timing_change else ""
            best_marker = " ★" if v.version == self.mem.best_version else ""
            logging.info(f"  V{v.version} [{status}] {timing_str}{delta}{best_marker} - {v.intended_change[:50]}...")
        
        logging.info("\nKey Insights:")
        for insight in self.mem.insights[-5:]:
            logging.info(f"  • {insight}")


# --------------------------------------------------------------------------- #
# 9. ENTRY POINT
# --------------------------------------------------------------------------- #

OPTIMIZER_TYPE = "simple"

def run_experiment(task: str, model: str, suffix: str, iters: int, profile: bool):
    if model not in MODEL_CONFIGS:
        logging.error(f"Unknown model: {model}. Use --list-models")
        return
    
    cfg = TASK_CONFIGS[task]
    mcfg = MODEL_CONFIGS[model]
    
    if not os.getenv(mcfg["api_key_env"]):
        logging.error(f"{mcfg['api_key_env']} not set!")
        return
    
    out = f"./{task}_{mcfg['short_name']}_{OPTIMIZER_TYPE}_{suffix}"
    if not profile:
        out += "_noprofile"
    
    config = OptimizationConfig(
        task=task,
        leaderboard=cfg["leaderboard"],
        task_description=cfg["description"],
        task_interface=cfg["interface"],
        model_name=mcfg["name"],
        model_short=mcfg["short_name"],
        api_base=mcfg["api_base"],
        api_key_env=mcfg["api_key_env"],
        optimizer_type=OPTIMIZER_TYPE,
        max_iterations=iters,
        use_profile=profile,
        output_dir=out
    )
    
    GPUKernelOptimizationAgent(config).run()


def list_models():
    print("\n" + "="*70)
    print("AVAILABLE MODELS (v8)")
    print("="*70)
    
    for tier in ["tier1", "tier2", "tier3"]:
        tier_name = {"tier1": "TIER 1 (Flagship)", "tier2": "TIER 2 (Excellent)", "tier3": "TIER 3 (Good)"}[tier]
        print(f"\n{tier_name}:")
        print("-" * 50)
        for model_key in MODEL_TIERS[tier]:
            m = MODEL_CONFIGS[model_key]
            print(f"  {model_key:18} : {m['description']}")
    
    print("\n" + "="*70)
    print(f"Total: {len(ALL_MODELS)} models")
    print("="*70 + "\n")


def main():
    p = argparse.ArgumentParser(
        description='GPU Optimizer v8 (Simple) - ROBUST EXTRACTION with auto-retry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v8 Key Features:
  - Multi-pattern JSON/code extraction (handles various LLM output formats)
  - Auto-retry mechanism (up to 3 retries with simpler prompts)
  - Full history preserved (LLM sees all versions)

Examples:
  python3 gpu_kernel_optimizer_simple.py --task flashattention --model gemini3-pro --suffix v1
  python3 gpu_kernel_optimizer_simple.py --task flashattention --all-models --suffix v1
  python3 gpu_kernel_optimizer_simple.py --list-models
        """
    )
    
    p.add_argument('--task', choices=['histogram', 'flashattention'], default='histogram')
    p.add_argument('--model', default='gpt5', help='Model to use')
    p.add_argument('--suffix', default='v1')
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--all-models', action='store_true', help='Run all models')
    p.add_argument('--tier', choices=['tier1', 'tier2', 'tier3'], help='Run models in tier')
    p.add_argument('--list-models', action='store_true')
    
    args = p.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimizer.log', 'w'),
            logging.StreamHandler()
        ]
    )
    
    if args.list_models:
        list_models()
        return
    
    # Determine models to run
    if args.all_models:
        models = ALL_MODELS
    elif args.tier:
        models = MODEL_TIERS[args.tier]
    else:
        models = [args.model]
    
    # Run experiments
    for model in models:
        logging.info(f"\n{'#'*60}")
        logging.info(f"# {args.task} + {model}")
        logging.info(f"{'#'*60}\n")
        try:
            run_experiment(args.task, model, args.suffix, args.iterations, not args.no_profile)
        except Exception as e:
            logging.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(5)


if __name__ == "__main__":
    main()
