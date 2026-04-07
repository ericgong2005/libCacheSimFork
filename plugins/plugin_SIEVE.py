from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
    """
    SIEVE eviction policy:
      - One insertion-ordered queue (implemented as doubly-linked list)
      - One 'hand' pointer walking from tail toward head
      - One visited bit per node
      - On hit: mark visited
      - On eviction: walk hand; if visited => clear and move; else evict

    This is the canonical SIEVE described in SIEVE writeups.
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "visited")

        def __init__(self, obj_id: int):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.visited = False

    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        # Required by free_hook
        self.queue: dict[int, FifoCache._Node] = {}

        # Sentinels
        self._head = self._Node(-1)  # MRU / insertion end
        self._tail = self._Node(-2)  # LRU / scan end
        self._head.next = self._tail
        self._tail.prev = self._head

        self._hand: FifoCache._Node | None = None

    def _insert_head(self, node: _Node):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def on_hit(self, req: Request):
        node = self.queue.get(req.obj_id)
        if node is not None:
            node.visited = True

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        node = self._Node(obj_id)
        self.queue[obj_id] = node
        self._insert_head(node)
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None

    def evict(self, req: Request):
        if not self.queue:
            return 0
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None
            if self._hand is None:
                return 0

        node = self._hand
        while True:
            if node is self._head:
                node = self._tail.prev
                continue

            if node.visited:
                node.visited = False
                node = node.prev
                continue

            # Evict node
            victim = node
            # Advance hand before removing
            self._hand = victim.prev if victim.prev is not self._head else self._tail.prev
            vid = victim.obj_id
            self._remove_node(victim)
            self.queue.pop(vid, None)
            if not self.queue:
                self._hand = None
            return vid

    def on_remove(self, obj_id: int):
        node = self.queue.pop(obj_id, None)
        if node is None:
            return
        if self._hand is node:
            self._hand = node.prev if node.prev is not self._head else self._tail.prev
            if self._hand is self._head:
                self._hand = None
        self._remove_node(node)
        if not self.queue:
            self._hand = None



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
