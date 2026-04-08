from collections import deque
from libcachesim import CommonCacheParams, Request


class GDSFDecayReinsertCache:
    """
    GDSF with:
      - exponential frequency decay
      - bounded ghost list of recently evicted IDs
      - ghost-hit reinsertion boost

    Motivation:
      Plain GDSF is good when size matters.
      Decay helps when popularity shifts over time.
      Ghost reinsert helps when evicted objects recur soon after eviction.

    Score:
        H(x) = L + freq(x) / size(x)

    On hit:
        freq = freq * decay + 1

    On miss:
        normal insert:      freq = 1.0
        ghost-hit insert:   freq = ghost_boost

    On eviction:
        evict smallest H
        set L = H(victim)
        add victim to ghost
    """

    def __init__(
        self,
        cache_size: int,
        decay: float = 0.92,
        ghost_boost: float = 2.5,
        ghost_max: int = 100_000,
    ):
        import heapq

        self.cache_size = cache_size
        self.decay = decay
        self.ghost_boost = ghost_boost

        # obj_id -> (size, freq, H, ver)
        self.queue: dict[int, tuple[int, float, float, int]] = {}
        self.heap: list[tuple[float, int, int]] = []

        self.L = 0.0
        self._ver = 0
        self._heapq = heapq

        # Ghost history
        self.G_q = deque()
        self.G_s: set[int] = set()
        self.G_max = ghost_max

    def _ghost_add(self, obj_id: int):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        while len(self.G_s) > self.G_max and self.G_q:
            old = self.G_q.popleft()
            self.G_s.discard(old)

    def _push(self, obj_id: int, size: int, freq: float):
        self._ver += 1
        H = self.L + (freq / max(size, 1))
        self.queue[obj_id] = (size, freq, H, self._ver)
        self._heapq.heappush(self.heap, (H, self._ver, obj_id))

    def on_hit(self, req: Request):
        rec = self.queue.get(req.obj_id)
        if rec is None:
            return

        size, freq, _, _ = rec
        size = req.obj_size or size
        freq = freq * self.decay + 1.0
        self._push(req.obj_id, size, freq)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        size = req.obj_size

        if obj_id in self.G_s:
            self.G_s.discard(obj_id)
            self._push(obj_id, size, self.ghost_boost)
        else:
            self._push(obj_id, size, 1.0)

    def evict(self, req: Request):
        if not self.queue:
            return 0

        while self.heap:
            H, ver, obj_id = self.heap[0]
            rec = self.queue.get(obj_id)

            if rec is None:
                self._heapq.heappop(self.heap)
                continue

            _, _, H_cur, ver_cur = rec
            if ver != ver_cur or H != H_cur:
                self._heapq.heappop(self.heap)
                continue

            self._heapq.heappop(self.heap)
            self.L = H
            self.queue.pop(obj_id, None)
            self._ghost_add(obj_id)
            return obj_id

        # Rare fallback
        obj_id = next(iter(self.queue.keys()))
        self.queue.pop(obj_id, None)
        self._ghost_add(obj_id)
        return obj_id

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    return GDSFDecayReinsertCache(cs, decay=0.90, ghost_boost=2.0, ghost_max=150_000)
    # return GDSFDecayReinsertCache(cs, decay=0.90, ghost_boost=3.0, ghost_max=50_000)



def hit_hook(data: GDSFDecayReinsertCache, req: Request):
    data.on_hit(req)


def miss_hook(data: GDSFDecayReinsertCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: GDSFDecayReinsertCache, req: Request):
    return data.evict(req)


def remove_hook(data: GDSFDecayReinsertCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: GDSFDecayReinsertCache):
    data.queue.clear()


if __name__ == "__main__":
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    plugin_cache = PluginCache(
        cache_size=1024 * 1024,
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
        cache_name="gdsf_decay_ghost",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio: {byte_miss_ratio:.4f}")
