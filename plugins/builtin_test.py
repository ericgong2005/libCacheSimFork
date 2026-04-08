from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

from libcachesim import (
    CommonCacheParams,
    Request,
    CacheBase,
    LRU,
    FIFO,
    LFU,
    ARC,
    S3FIFO,
    Sieve,
    LIRS,
    GDSF,
)

def init_hook(common_cache_params: CommonCacheParams):
    return LIRS(common_cache_params.cache_size)

def hit_hook(data, req: Request):
    data.get(req)

def miss_hook(data, req: Request):
    if data.can_insert(req):
        data.insert(req)

def eviction_hook(data, req: Request):
    victim = data.evict(req)
    return victim.obj_id if victim is not None else 0

def remove_hook(data, obj_id: int):
    data.remove(obj_id)

def free_hook(data):
    pass

if __name__ == "__main__":
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    plugin_fifo_cache = PluginCache(
        cache_size=1024 * 1024,
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
        cache_name="fifo",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_fifo_cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio: {byte_miss_ratio:.4f}")
