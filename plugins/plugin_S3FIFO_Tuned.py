from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    Tuned S3-FIFO variant:
      - Larger probation region (small_ratio=0.20)
      - Promote after 1 re-reference (threshold=1)
      - Slightly stronger main counter (max=3)

    Intended to improve hit rate on workloads with moderate reuse bursts
    by moving useful items into M earlier.
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "size", "cnt")

        def __init__(self, obj_id: int, size: int, cnt: int = 0):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.size = size
            self.cnt = cnt

    class _DLL:
        __slots__ = ("head", "tail", "bytes", "nodes")

        def __init__(self):
            self.head = FifoCache._Node(-1, 0)
            self.tail = FifoCache._Node(-2, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.bytes = 0
            self.nodes: dict[int, FifoCache._Node] = {}

        def empty(self) -> bool:
            return self.head.next is self.tail

        def push_head(self, node: "FifoCache._Node"):
            node.next = self.head.next
            node.prev = self.head
            self.head.next.prev = node
            self.head.next = node
            self.nodes[node.obj_id] = node
            self.bytes += node.size

        def pop_tail(self) -> "FifoCache._Node | None":
            if self.empty():
                return None
            node = self.tail.prev
            self.remove(node.obj_id)
            return node

        def remove(self, obj_id: int) -> "FifoCache._Node | None":
            node = self.nodes.pop(obj_id, None)
            if node is None:
                return None
            node.prev.next = node.next
            node.next.prev = node.prev
            self.bytes -= node.size
            node.prev = node.next = None
            return node

    def __init__(self, cache_size: int):
        from collections import deque

        self.cache_size = cache_size
        self.S = self._DLL()
        self.M = self._DLL()

        self.G_q = deque()
        self.G_s: set[int] = set()
        self.G_max = 200_000

        self.small_ratio = 0.20
        self.small_target = int(self.cache_size * self.small_ratio)
        self.promote_threshold = 1
        self.main_cnt_max = 3

        self.queue: dict[int, int] = {}

    def _ghost_add(self, obj_id: int):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        while len(self.G_s) > self.G_max and self.G_q:
            old = self.G_q.popleft()
            self.G_s.discard(old)

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.S.nodes:
            node = self.S.nodes[obj_id]
            node.cnt += 1
            if node.cnt >= self.promote_threshold:
                moved = self.S.remove(obj_id)
                if moved is not None:
                    moved.cnt = min(2, self.main_cnt_max)
                    self.M.push_head(moved)
        elif obj_id in self.M.nodes:
            node = self.M.nodes[obj_id]
            node.cnt = min(self.main_cnt_max, node.cnt + 1)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        size = req.obj_size
        if obj_id in self.G_s:
            self.G_s.discard(obj_id)
            node = self._Node(obj_id, size=size, cnt=2)
            self.M.push_head(node)
        else:
            node = self._Node(obj_id, size=size, cnt=0)
            self.S.push_head(node)
        self.queue[obj_id] = size

    def evict(self, req: Request):
        if not self.queue:
            return 0

        # If probation is relatively large, still prefer evicting probation first.
        prefer_S = (not self.S.empty()) and (self.S.bytes > self.small_target or self.M.empty())

        if prefer_S:
            while not self.S.empty():
                cand = self.S.pop_tail()
                if cand is None:
                    break
                if cand.cnt >= self.promote_threshold:
                    cand.cnt = min(2, self.main_cnt_max)
                    self.M.push_head(cand)
                    continue
                vid = cand.obj_id
                self.queue.pop(vid, None)
                self._ghost_add(vid)
                return vid

        while not self.M.empty():
            cand = self.M.pop_tail()
            if cand is None:
                break
            if cand.cnt > 0:
                cand.cnt -= 1
                self.M.push_head(cand)
                continue
            vid = cand.obj_id
            self.queue.pop(vid, None)
            return vid

        vid = next(iter(self.queue.keys()))
        self.on_remove(vid)
        return vid

    def on_remove(self, obj_id: int):
        if obj_id in self.S.nodes:
            self.S.remove(obj_id)
        if obj_id in self.M.nodes:
            self.M.remove(obj_id)
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
