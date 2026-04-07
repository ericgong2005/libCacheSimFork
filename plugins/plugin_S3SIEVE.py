from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    S3-SIEVE hybrid:
      - Probation queue S: FIFO-ish with promotion threshold
      - Protected/main M: SIEVE (lazy promotion), aiming to preserve hot items without LRU reordering

    Motivation:
      If hits are frequent, LRU-style reordering can be expensive (and sometimes unnecessary for hit rate);
      SIEVE keeps hit-path minimal (set a bit) and does most of its work at eviction time.

    This matches common discussions of "plugging SIEVE into S3-FIFO-like architectures."
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "size", "cnt", "visited")

        def __init__(self, obj_id: int, size: int, cnt: int = 0, visited: bool = False):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.size = size
            self.cnt = cnt
            self.visited = visited

    class _DLL:
        __slots__ = ("head", "tail", "bytes", "nodes")

        def __init__(self):
            self.head = FifoCache._Node(-1, 0)
            self.tail = FifoCache._Node(-2, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.bytes = 0
            self.nodes: dict[int, FifoCache._Node] = {}

        def empty(self):
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

        # Main SIEVE structure
        self.M_nodes: dict[int, FifoCache._Node] = {}
        self.M_head = self._Node(-10, 0)
        self.M_tail = self._Node(-11, 0)
        self.M_head.next = self.M_tail
        self.M_tail.prev = self.M_head
        self.M_bytes = 0
        self.M_hand: FifoCache._Node | None = None

        # Ghost
        self.G_q = deque()
        self.G_s: set[int] = set()
        self.G_max = 200_000

        # Params
        self.small_ratio = 0.10
        self.small_target = int(self.cache_size * self.small_ratio)
        self.promote_threshold = 2

        # Boilerplate free_hook target
        self.queue: dict[int, int] = {}

    def _ghost_add(self, obj_id: int):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        while len(self.G_s) > self.G_max and self.G_q:
            old = self.G_q.popleft()
            self.G_s.discard(old)

    # ----- Main SIEVE list operations -----
    def _M_insert_head(self, node: _Node):
        node.next = self.M_head.next
        node.prev = self.M_head
        self.M_head.next.prev = node
        self.M_head.next = node
        self.M_nodes[node.obj_id] = node
        self.M_bytes += node.size
        if self.M_hand is None:
            self.M_hand = self.M_tail.prev if self.M_tail.prev is not self.M_head else None

    def _M_remove_node(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.M_bytes -= node.size
        self.M_nodes.pop(node.obj_id, None)
        node.prev = node.next = None

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.S.nodes:
            node = self.S.nodes[obj_id]
            node.cnt += 1
            if node.cnt >= self.promote_threshold:
                moved = self.S.remove(obj_id)
                if moved is not None:
                    moved.visited = True
                    self._M_insert_head(moved)
        elif obj_id in self.M_nodes:
            self.M_nodes[obj_id].visited = True

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        size = req.obj_size

        if obj_id in self.G_s:
            self.G_s.discard(obj_id)
            node = self._Node(obj_id, size=size, visited=True)
            self._M_insert_head(node)
        else:
            node = self._Node(obj_id, size=size, cnt=0)
            self.S.push_head(node)

        self.queue[obj_id] = size

    def _evict_from_M(self) -> int:
        if not self.M_nodes:
            return 0
        if self.M_hand is None:
            self.M_hand = self.M_tail.prev if self.M_tail.prev is not self.M_head else None
            if self.M_hand is None:
                return 0

        node = self.M_hand
        while True:
            if node is self.M_head:
                node = self.M_tail.prev
                continue
            if node.visited:
                node.visited = False
                node = node.prev
                continue

            victim = node
            self.M_hand = victim.prev if victim.prev is not self.M_head else self.M_tail.prev
            vid = victim.obj_id
            self._M_remove_node(victim)
            self.queue.pop(vid, None)
            if not self.M_nodes:
                self.M_hand = None
            return vid

    def evict(self, req: Request):
        if not self.queue:
            return 0

        # Prefer evicting from S for quick demotion, unless S is small and M has content.
        prefer_S = (not self.S.empty()) and (self.S.bytes > self.small_target or not self.M_nodes)

        if prefer_S:
            while not self.S.empty():
                cand = self.S.pop_tail()
                if cand is None:
                    break
                if cand.cnt >= self.promote_threshold:
                    cand.visited = True
                    self._M_insert_head(cand)
                    continue
                vid = cand.obj_id
                self.queue.pop(vid, None)
                self._ghost_add(vid)
                return vid

        # Evict from SIEVE main
        vid = self._evict_from_M()
        if vid:
            return vid

        # Fallback
        vid = next(iter(self.queue.keys()))
        self.on_remove(vid)
        return vid

    def on_remove(self, obj_id: int):
        if obj_id in self.S.nodes:
            self.S.remove(obj_id)
        if obj_id in self.M_nodes:
            node = self.M_nodes.get(obj_id)
            if node is not None:
                if self.M_hand is node:
                    self.M_hand = node.prev if node.prev is not self.M_head else self.M_tail.prev
                self._M_remove_node(node)
                if not self.M_nodes:
                    self.M_hand = None
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
