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
