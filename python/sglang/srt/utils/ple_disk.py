# Copyright 2026 SGLang Team
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
# ==============================================================================

IORING_MAX_ENTRIES = 32768
PLE_DISK_MAX_PREFILL_BUFFER_TOKENS = 65536
PLE_DISK_DEFAULTS = {
    "ple_disk_dir": None,
    "ple_disk_hot_cache_gb": 8.0,
    "ple_disk_hot_frequency_file": None,
    "ple_disk_dynamic_cache_gb": 2.0,
    "ple_disk_prefill_buffer_tokens": 8192,
    "ple_disk_prefill_read_pages": 128,
    "ple_disk_max_read_pages": None,
    "ple_disk_stats_log_interval": 0,
    "ple_disk_cleanup_generations": True,
    "ple_disk_max_prefill_chunk_tokens": 0,
}


def resolve_ple_storage(config, default=None):
    """Return the resolved PLE storage mode from a runtime config object."""
    storage = getattr(config, "ple_storage", None)
    return storage if storage is not None else default


def resolve_model_runner_ple_storage(model_runner, default=None):
    """Resolve storage from the published runner config, then its HF config."""
    storage = resolve_ple_storage(getattr(model_runner, "server_args", None))
    if storage is not None:
        return storage
    model_config = getattr(model_runner, "model_config", None)
    return resolve_ple_storage(getattr(model_config, "hf_text_config", None), default)


def validate_max_read_pages(value: int) -> int:
    value = int(value)
    if not 1 <= value <= IORING_MAX_ENTRIES:
        raise ValueError(
            "--ple-disk-max-read-pages must be between 1 and "
            f"{IORING_MAX_ENTRIES}, got {value}"
        )
    return value
