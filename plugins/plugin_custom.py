
from collections import deque
from libcachesim import CommonCacheParams, Request

SMALL_RATIO  = 0.10   # fraction of cache reserved for S
GHOST_RATIO  = 0.10   # ghost queue
MAX_FREQ     = 3      # counter cap

class S3FifoCache:

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.s_max = max(1, int(cache_size * SMALL_RATIO))
        self.m_max = cache_size - self.s_max
        self.g_max = max(1, int(cache_size * GHOST_RATIO))

        self.s_used = 0
        self.m_used = 0

        self.s_queue: deque[int] = deque()
        self.m_queue: deque[int] = deque()

        self.ghost: dict[int, None] = {}

        self.freq:     dict[int, int] = {}
        self.obj_size: dict[int, int] = {}


    def add_to_ghost(self, obj_id: int) -> None:
        if obj_id in self.ghost:
            return
        if len(self.ghost) >= self.g_max:
            oldest = next(iter(self.ghost))
            del self.ghost[oldest]
        self.ghost[obj_id] = None

    def evict_from_s(self) -> int:
        obj_id = self.s_queue.popleft()
        size   = self.obj_size.pop(obj_id, 0)
        f      = self.freq.pop(obj_id, 0)
        self.s_used -= size

        if f >= 1:
            self.make_room_in_m(size)
            self.m_queue.append(obj_id)
            self.m_used += size
            self.obj_size[obj_id] = size
            self.freq[obj_id] = min(f, MAX_FREQ)
            return 0

        self.add_to_ghost(obj_id)
        return obj_id

    def evict_from_m(self) -> int:
        obj_id = self.m_queue.popleft()
        size   = self.obj_size.pop(obj_id, 0)
        self.freq.pop(obj_id, None)
        self.m_used -= size
        return obj_id

    def make_room_in_m(self, needed: int) -> None:
        while self.m_used + needed > self.m_max and self.m_queue:
            self.evict_from_m()

    def on_hit(self, req: Request) -> None:
        obj_id = req.obj_id
        if obj_id in self.freq:
            self.freq[obj_id] = min(self.freq[obj_id] + 1, MAX_FREQ)

    def on_miss(self, req: Request) -> None:
        obj_id = req.obj_id
        size   = req.obj_size

        if size > self.cache_size:
            return

        if obj_id in self.ghost:
            del self.ghost[obj_id]
            self.make_room_in_m(size)
            self.m_queue.append(obj_id)
            self.m_used += size
            self.obj_size[obj_id] = size
            self.freq[obj_id] = 0
        else:
            while self.s_used + size > self.s_max and self.s_queue:
                self.evict_from_s()
            while self.s_used + size > self.s_max:
                self.evict_from_m()
            self.s_queue.append(obj_id)
            self.s_used += size
            self.obj_size[obj_id] = size
            self.freq[obj_id] = 0

    def evict(self, req: Request) -> int:
        while self.s_queue:
            evicted = self.evict_from_s()
            if evicted != 0:
                return evicted

        if self.m_queue:
            return self.evict_from_m()

        return 0

    def on_remove(self, obj_id: int) -> None:
        size = self.obj_size.pop(obj_id, None)
        if size is None:
            return

        self.freq.pop(obj_id, None)

        try:
            self.s_queue.remove(obj_id)
            self.s_used -= size
            return
        except ValueError:
            pass

        try:
            self.m_queue.remove(obj_id)
            self.m_used -= size
        except ValueError:
            pass

def init_hook(common_cache_params: CommonCacheParams) -> S3FifoCache:
    return S3FifoCache(common_cache_params.cache_size)


def hit_hook(data: S3FifoCache, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: S3FifoCache, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: S3FifoCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: S3FifoCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: S3FifoCache) -> None:
    data.s_queue.clear()
    data.m_queue.clear()
    data.ghost.clear()
    data.freq.clear()
    data.obj_size.clear()

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
        cache_name="s3fifo",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
