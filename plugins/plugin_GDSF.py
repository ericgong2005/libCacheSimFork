from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    GDSF-like (GreedyDual Size-Frequency) eviction for variable-size objects.

    Core idea:
      Maintain a key H(x) = L + freq(x)/size(x) for each object.
      Evict smallest H; on eviction set global L = H(victim) (aging).

    This is a widely used pattern for byte-constrained object caches.

    Implementation:
      - Min-heap of (H, version, obj_id)
      - Dict obj_id -> (size, freq, H, version)
      - Lazy deletion via versioning
    """

    def __init__(self, cache_size: int):
        import heapq

        self.cache_size = cache_size
        self.queue: dict[int, tuple[int, int, float, int]] = {}  # obj_id -> (size, freq, H, ver)
        self.heap: list[tuple[float, int, int]] = []  # (H, ver, obj_id)
        self.L = 0.0
        self._ver = 0
        self._heapq = heapq

    def _push(self, obj_id: int, size: int, freq: int):
        self._ver += 1
        H = self.L + (freq / max(size, 1))
        self.queue[obj_id] = (size, freq, H, self._ver)
        self._heapq.heappush(self.heap, (H, self._ver, obj_id))

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        rec = self.queue.get(obj_id)
        if rec is None:
            return
        size, freq, _, _ = rec
        # If size changes (rare), update.
        size = req.obj_size or size
        freq = freq + 1
        self._push(obj_id, size, freq)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        self._push(req.obj_id, req.obj_size, 1)

    def evict(self, req: Request):
        if not self.queue:
            return 0

        # Pop until we find a live record
        while self.heap:
            H, ver, obj_id = self.heap[0]
            rec = self.queue.get(obj_id)
            if rec is None:
                self._heapq.heappop(self.heap)
                continue
            _, _, H_current, ver_current = rec
            if ver != ver_current or H != H_current:
                self._heapq.heappop(self.heap)
                continue

            # Found victim
            self._heapq.heappop(self.heap)
            self.L = H  # aging
            self.queue.pop(obj_id, None)
            return obj_id

        # Fallback (should be rare)
        obj_id = next(iter(self.queue.keys()))
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
