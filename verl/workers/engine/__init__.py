# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .base import BaseEngine, EngineRegistry
from .fsdp import FSDPEngine, FSDPEngineWithLMHead

__all__ = ["BaseEngine", "EngineRegistry", "FSDPEngine", "FSDPEngineWithLMHead"]

# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None

# [MODIFIED START] Safe import for AMD/ROCm environment compatibility
# Original code only caught ImportError. We catch Exception because transformer_engine
# on AMD might raise RuntimeError (No HIP GPUs available) during import.
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
