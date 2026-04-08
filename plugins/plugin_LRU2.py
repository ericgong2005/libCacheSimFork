from collections import deque
from libcachesim import CommonCacheParams, Request

class LRU2Cache:
    """
    LRU-2 (a simple LRU-K with K=2):

    Eviction priority is based on the time of the 2nd most recent reference.
    - Objects referenced only once have last2 = -inf, so they are evicted first.
    - Objects with repeated reads get better protection even if their most recent access isn't super recent.

    This can be very effective on scan + hotset traces where single-touch items flood the cache.
    """

    def __init__(self, cache_size: int):
        import heapq

        self.cache_size = cache_size
        self._heapq = heapq

        # Required by free_hook
        # obj_id -> (size, last1, last2, ver)
        self.queue: dict[int, tuple[int, float, float, int]] = {}
        self._heap: list[tuple[float, int, int]] = []  # (last2, ver, obj_id)
        self._ver = 0

    @staticmethod
    def _time(req: Request) -> float:
        # Prefer clock_time; if absent, fall back to vtime; else monotonic counter not available here.
        t = getattr(req, "clock_time", None)
        if t is None:
            t = getattr(req, "vtime", None)
        try:
            return float(t) if t is not None else 0.0
        except Exception:
            return 0.0

    def _push(self, obj_id: int, size: int, last1: float, last2: float):
        self._ver += 1
        self.queue[obj_id] = (size, last1, last2, self._ver)
        self._heapq.heappush(self._heap, (last2, self._ver, obj_id))

    def on_hit(self, req: Request):
        rec = self.queue.get(req.obj_id)
        if rec is None:
            return
        size, last1, last2, _ = rec
        t = self._time(req)
        # shift timestamps: last2 <- last1, last1 <- now
        self._push(req.obj_id, size, t, last1)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        t = self._time(req)
        # last2 = -inf (represented as a very small number) for "only seen once"
        self._push(req.obj_id, req.obj_size, t, float("-inf"))

    def evict(self, req: Request):
        if not self.queue:
            return 0

        while self._heap:
            last2, ver, obj_id = self._heap[0]
            rec = self.queue.get(obj_id)
            if rec is None:
                self._heapq.heappop(self._heap)
                continue
            _, _, last2_cur, ver_cur = rec
            if ver != ver_cur or last2 != last2_cur:
                self._heapq.heappop(self._heap)
                continue

            self._heapq.heappop(self._heap)
            self.queue.pop(obj_id, None)
            return obj_id

        # fallback
        obj_id = next(iter(self.queue.keys()))
        self.queue.pop(obj_id, None)
        return obj_id

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams):
    return LRU2Cache(common_cache_params.cache_size)


def hit_hook(data, req: Request):
    data.on_hit(req)


def miss_hook(data, req: Request):
    data.on_miss(req)


def eviction_hook(data, req: Request):
    return data.evict(req)


def remove_hook(data, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data):
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
