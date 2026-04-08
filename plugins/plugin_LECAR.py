from collections import deque
from libcachesim import CommonCacheParams, Request
import math

class LECARCache:
    """
    LeCaR: Learning Cache Replacement.

    Uses regret minimization to dynamically blend LRU and LFU.
    Maintains:
      - An LRU eviction policy
      - An LFU eviction policy (min-heap by frequency)
      - A weight w in [0,1]: probability of using LRU vs LFU for eviction
      - Ghost lists for each policy to learn which would have been better

    On ghost hit from LRU's ghost -> LFU was right -> decrease w (favor LFU)
    On ghost hit from LFU's ghost -> LRU was right -> increase w (favor LRU)

    Different from ARC because:
    - ARC adapts partition SIZE between recency/frequency lists
    - LeCaR adapts the PROBABILITY of choosing which policy to evict from
    - LeCaR uses multiplicative weight update (exponential learning)
    """

    def __init__(self, cache_size: int, learning_rate: float = 0.45, discount: float = 0.005):
        import heapq
        import random

        self.cache_size = cache_size
        self.lr = learning_rate
        self.discount = discount
        self._heapq = heapq
        self._random = random

        self.w = 0.5

        # LRU: OrderedDict, LRU at front
        self.lru_order: OrderedDict[int, int] = OrderedDict()

        # LFU: dict + lazy heap
        self.freq: dict[int, int] = {}
        self.heap: list[tuple[int, int, int]] = []
        self._ver = 0

        # Ghost lists
        self.ghost_lru: OrderedDict[int, None] = OrderedDict()
        self.ghost_lfu: OrderedDict[int, None] = OrderedDict()
        self.ghost_max = 100_000

        self.queue: dict[int, int] = {}
        self._time = 0

    def _lfu_push(self, obj_id: int):
        f = self.freq.get(obj_id, 0)
        self._ver += 1
        self._heapq.heappush(self.heap, (f, self._ver, obj_id))

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id not in self.queue:
            return
        if obj_id in self.lru_order:
            self.lru_order.move_to_end(obj_id)
        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1
        self._lfu_push(obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        sz = req.obj_size
        self._time += 1

        # Learn from ghost hits
        if obj_id in self.ghost_lru:
            self.ghost_lru.pop(obj_id)
            d = math.pow(1 - self.discount, self._time)
            self.w = max(0.001, self.w * math.exp(-self.lr * d))

        elif obj_id in self.ghost_lfu:
            self.ghost_lfu.pop(obj_id)
            d = math.pow(1 - self.discount, self._time)
            self.w = min(0.999, 1.0 - (1.0 - self.w) * math.exp(-self.lr * d))

        self.queue[obj_id] = sz
        self.lru_order[obj_id] = sz
        self.freq[obj_id] = 1
        self._lfu_push(obj_id)

    def evict(self, req: Request):
        if not self.queue:
            return 0

        if self._random.random() < self.w:
            vid = self._evict_lru()
            if vid is not None:
                self.ghost_lru[vid] = None
                if len(self.ghost_lru) > self.ghost_max:
                    self.ghost_lru.popitem(last=False)
                return vid
            vid = self._evict_lfu()
            if vid is not None:
                return vid
        else:
            vid = self._evict_lfu()
            if vid is not None:
                self.ghost_lfu[vid] = None
                if len(self.ghost_lfu) > self.ghost_max:
                    self.ghost_lfu.popitem(last=False)
                return vid
            vid = self._evict_lru()
            if vid is not None:
                return vid

        vid = next(iter(self.queue))
        self.queue.pop(vid)
        return vid

    def _evict_lru(self):
        while self.lru_order:
            vid, vsz = self.lru_order.popitem(last=False)
            if vid in self.queue:
                self.queue.pop(vid)
                self.freq.pop(vid, None)
                return vid
        return None

    def _evict_lfu(self):
        while self.heap:
            f, ver, obj_id = self.heap[0]
            if obj_id not in self.queue:
                self._heapq.heappop(self.heap)
                continue
            cur_f = self.freq.get(obj_id, 0)
            if cur_f != f:
                self._heapq.heappop(self.heap)
                continue
            self._heapq.heappop(self.heap)
            self.queue.pop(obj_id)
            self.lru_order.pop(obj_id, None)
            self.freq.pop(obj_id, None)
            return obj_id
        return None

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)
        self.lru_order.pop(obj_id, None)
        self.freq.pop(obj_id, None)

def init_hook(common_cache_params: CommonCacheParams):
    return LECARCache(common_cache_params.cache_size)


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
