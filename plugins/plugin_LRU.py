from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    LRU baseline using ordered dict semantics.

    Policy:
      - On hit, move to MRU position.
      - Evict least-recently used.

    Notes:
      - Uses only built-ins; no extra imports required.
      - self.queue is the ordered mapping: obj_id -> obj_size (size stored for potential debug).
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.queue: dict[int, int] = {}  # insertion-ordered in Python 3.7+

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        # Move to MRU: pop and re-insert (dict keeps insertion order).
        try:
            sz = self.queue.pop(obj_id)
        except KeyError:
            # Defensive: simulator says hit, but we're missing it in policy state.
            return
        self.queue[obj_id] = sz

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        # Insert as MRU.
        self.queue[req.obj_id] = req.obj_size

    def evict(self, req: Request):
        # Evict LRU = first item
        if not self.queue:
            return 0
        obj_id, _ = next(iter(self.queue.items()))
        self.queue.pop(obj_id, None)
        return obj_id

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams):
    return FifoCache(common_cache_params.cache_size)


def hit_hook(data: FifoCache, req: Request):
    data.on_hit(req)


def miss_hook(data: FifoCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: FifoCache, req: Request):
    return data.evict(req)


def remove_hook(data: FifoCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: FifoCache):
    data.queue.clear()


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
