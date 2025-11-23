# Verl Codebase Modifications Record

This document tracks modifications made to the original Verl codebase to support AMD GPU environments and other custom requirements.

## 1. Safe Import for Megatron Engine
**File:** `verl/workers/engine/__init__.py`
**Reason:** The original code strictly imports `MegatronEngine`, which depends on `transformer_engine`. On AMD GPUs (ROCm), `transformer_engine` may fail to initialize (looking for HIP GPUs), causing the entire program to crash even if Megatron is not used.
**Change:** Wrapped the import in a broad `try-except` block to catch `Exception` instead of just `ImportError`, allowing the program to proceed if Megatron fails to load.

```python
# Before
try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead"]
except ImportError:
    MegatronEngine = None
    MegatronEngineWithLMHead = None

# After
# [MODIFIED START] Safe import for AMD/ROCm environment compatibility
try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead"]
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import MegatronEngine: {e}. If you are not using Megatron, please ignore this warning.")
    MegatronEngine = None
    MegatronEngineWithLMHead = None
# [MODIFIED END]
```

