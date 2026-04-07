from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    Belady MIN / OPT eviction policy (oracle):
      Evict the cached object whose next access is farthest in the future.

    Requires traces where Request.next_access_vtime is populated (e.g., ORACLE traces).
    libCacheSim exposes next_access_vtime for this purpose.

    Implementation:
      - Max-heap on next_access_vtime (store negative to use heapq as max-heap)
      - Lazy deletion using versioning
    """

    def __init__(self, cache_size: int):
        import heapq

        self.cache_size = cache_size
        self._heapq = heapq

        # Required by free_hook:
        self.queue: dict[int, tuple[int, int, float]] = {}  # obj_id -> (size, ver, next_time)

        self.heap: list[tuple[float, int, int]] = []  # (-next_time, ver, obj_id)
        self._ver = 0

    def _norm_next(self, nxt) -> float:
        # Treat missing/invalid as "infinite" future. Many oracle traces use -1 for "never again".
        if nxt is None:
            return float("inf")
        try:
            v = float(nxt)
        except Exception:
            return float("inf")
        if v < 0:
            return float("inf")
        return v

    def _push(self, obj_id: int, size: int, next_time: float):
        self._ver += 1
        self.queue[obj_id] = (size, self._ver, next_time)
        self._heapq.heappush(self.heap, (-next_time, self._ver, obj_id))

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        rec = self.queue.get(obj_id)
        if rec is None:
            return
        size, _, _ = rec
        next_time = self._norm_next(getattr(req, "next_access_vtime", None))
        self._push(obj_id, size, next_time)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        size = req.obj_size
        next_time = self._norm_next(getattr(req, "next_access_vtime", None))
        self._push(obj_id, size, next_time)

    def evict(self, req: Request):
        if not self.queue:
            return 0

        while self.heap:
            neg_next, ver, obj_id = self.heap[0]
            rec = self.queue.get(obj_id)
            if rec is None:
                self._heapq.heappop(self.heap)
                continue
            _, ver_current, next_current = rec
            if ver != ver_current:
                self._heapq.heappop(self.heap)
                continue
            # Found valid farthest-next-use (largest next_time => smallest neg_next)
            self._heapq.heappop(self.heap)
            self.queue.pop(obj_id, None)
            return obj_id

        # Fallback
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
