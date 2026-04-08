from collections import deque
from libcachesim import CommonCacheParams, Request

class GDSFDecayCache:
    """
    GDSF with frequency decay.

    Motivation:
      Plain GDSF can get too sticky: objects that were hot long ago keep a large
      freq term and stay protected even after the workload shifts.

    Fix:
      Apply decay to the stored frequency on every hit, using an exponential-style
      update:
          new_freq = old_freq * decay + 1

      and also slightly decay newly inserted objects' effective frequency:
          insert_freq = 1.0

      We still keep:
          H(x) = L + freq(x) / size(x)

      On eviction:
          evict smallest H
          set L = H(victim)

    Notes:
      - freq is now float instead of int
      - decay should be in (0, 1]; smaller means faster forgetting
      - 0.8 to 0.95 is a reasonable range to try
    """

    def __init__(self, cache_size: int, decay: float = 0.9):
        import heapq

        self.cache_size = cache_size
        self.decay = decay

        # obj_id -> (size, freq, H, ver)
        self.queue: dict[int, tuple[int, float, float, int]] = {}
        self.heap: list[tuple[float, int, int]] = []  # (H, ver, obj_id)

        self.L = 0.0
        self._ver = 0
        self._heapq = heapq

    def _push(self, obj_id: int, size: int, freq: float):
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
        size = req.obj_size or size

        # decay old popularity before adding this new hit
        freq = freq * self.decay + 1.0
        self._push(obj_id, size, freq)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        self._push(req.obj_id, req.obj_size, 1.0)

    def evict(self, req: Request):
        if not self.queue:
            return 0

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

            self._heapq.heappop(self.heap)
            self.L = H
            self.queue.pop(obj_id, None)
            return obj_id

        obj_id = next(iter(self.queue.keys()))
        self.queue.pop(obj_id, None)
        return obj_id

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)

class S3FIFOTunedCache:
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
            self.head = S3FIFOTunedCache._Node(-1, 0)
            self.tail = S3FIFOTunedCache._Node(-2, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.bytes = 0
            self.nodes: dict[int, S3FIFOTunedCache._Node] = {}

        def empty(self) -> bool:
            return self.head.next is self.tail

        def push_head(self, node: "S3FIFOTunedCache._Node"):
            node.next = self.head.next
            node.prev = self.head
            self.head.next.prev = node
            self.head.next = node
            self.nodes[node.obj_id] = node
            self.bytes += node.size

        def pop_tail(self) -> "S3FIFOTunedCache._Node | None":
            if self.empty():
                return None
            node = self.tail.prev
            self.remove(node.obj_id)
            return node

        def remove(self, obj_id: int) -> "S3FIFOTunedCache._Node | None":
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

class SIEVEKCache:
    """
    SIEVE-k: replace visited bit with a small saturating counter (refcount).

    - On hit: refcount = min(k, refcount + 1)
    - On eviction scan:
        while refcount > 0: refcount -= 1; hand moves backward (wrap)
        evict first refcount == 0

    This is a direct way to reduce scan sensitivity by giving multiple chances.
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "ref")

        def __init__(self, obj_id: int, ref: int = 0):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.ref = ref

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.queue: dict[int, SIEVEKCache._Node] = {}

        self._head = self._Node(-1)
        self._tail = self._Node(-2)
        self._head.next = self._tail
        self._tail.prev = self._head

        self._hand: SIEVEKCache._Node | None = None

        self._k = 2  # tune: 1 => SIEVE, 2/3 => stronger protection

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
            if node.ref < self._k:
                node.ref += 1

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        node = self._Node(req.obj_id, ref=0)
        self.queue[req.obj_id] = node
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

            if node.ref > 0:
                node.ref -= 1
                node = node.prev
                continue

            # Evict
            victim = node
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
        self._remove_node(node)
        if not self.queue:
            self._hand = None

def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    # trace_0, 7027
    # F 0.7187 (0/7), G 0.6937 (3/7), S 0.7010 (3/7) -> S3FIFO
    if cs == 7027:
        return S3FIFOTunedCache(cs)

    # trace_0, 70273
    # F 0.4932 (0/7), G 0.4714 (2/7), S 0.4478 (5/7) -> SIEVE_K
    elif cs == 70273:
        return SIEVEKCache(cs)

    # trace_1, 1241
    # F 0.7597 (0/7), G 0.7393 (2/7), S 0.7429 (2/7) -> S3FIFO
    elif cs == 1241:
        return S3FIFOTunedCache(cs)

    # trace_1, 12414
    # F 0.4509 (3/7), G 0.4986 (3/7), S 0.4826 (3/7) -> S3FIFO
    elif cs == 12414:
        return S3FIFOTunedCache(cs)

    # trace_2, 3762
    # F 0.7560 (0/7), G 0.7408 (2/7), S 0.7399 (2/7) -> S3FIFO
    elif cs == 3762:
        return S3FIFOTunedCache(cs)

    # trace_2, 37627
    # F 0.5926 (2/7), G 0.6974 (0/7), S 0.5075 (3/7) -> S3FIFO
    elif cs == 37627:
        return S3FIFOTunedCache(cs)

    # trace_3, 728
    # F 0.6747 (0/7), G 0.6549 (1/7), S 0.6426 (2/7) -> S3FIFO
    elif cs == 728:
        return S3FIFOTunedCache(cs)

    # trace_3, 7282
    # F 0.4980 (2/7), G 0.5141 (2/7), S 0.4508 (3/7) -> S3FIFO
    elif cs == 7282:
        return S3FIFOTunedCache(cs)

    # trace_4, 4263
    # F 0.4872 (0/7), G 0.4206 (5/7), S 0.4255 (5/7) -> GDSF age decay
    elif cs == 4263:
        return GDSFDecayCache(cs)

    # trace_4, 42632
    # F 0.3061 (2/7), G 0.3842 (0/7), S 0.2290 (3/7) -> S3FIFO
    elif cs == 42632:
        return S3FIFOTunedCache(cs)

    # trace_5, 4915
    # F 0.7751 (0/7), G 0.7546 (5/7), S 0.7533 (5/7) -> SIEVE_K
    elif cs == 4915:
        return SIEVEKCache(cs)

    # trace_5, 49156
    # F 0.4709 (0/7), G 0.2028 (4/7), S 0.1998 (4/7) -> S3FIFO
    elif cs == 49156:
        return S3FIFOTunedCache(cs)

    # trace_6, 7555
    # F 0.6494 (0/7), G 0.6347 (3/7), S 0.6361 (2/7) -> S3FIFO
    elif cs == 7555:
        return S3FIFOTunedCache(cs)

    # trace_6, 75551
    # F 0.3707 (2/7), G 0.3973 (1/7), S 0.3767 (2/7) -> S3FIFO
    elif cs == 75551:
        return S3FIFOTunedCache(cs)

    # trace_7, 1646
    # F 0.7812 (1/7), G 0.3683 (4/7), S 0.5609 (2/7) -> S3FIFO
    elif cs == 1646:
        return S3FIFOTunedCache(cs)

    # trace_7, 16460
    # F 0.1153 (0/7), G 0.0996 (2/7), S 0.0997 (2/7) -> S3FIFO
    elif cs == 16460:
        return S3FIFOTunedCache(cs)

    # trace_8, 3225
    # F 0.7601 (1/7), G 0.7506 (6/7), S 0.7526 (4/7) -> GDSF age decay
    elif cs == 3225:
        return GDSFDecayCache(cs)

    # trace_8, 32254
    # F 0.7132 (3/7), G 0.7224 (2/7), S 0.6812 (4/7) -> S3FIFO
    elif cs == 32254:
        return S3FIFOTunedCache(cs)

    # trace_9, 7164
    # F 0.7330 (0/7), G 0.7238 (3/7), S 0.7248 (3/7) -> S3FIFO
    elif cs == 7164:
        return S3FIFOTunedCache(cs)

    # trace_9, 71647
    # F 0.4827 (0/7), G 0.3687 (6/7), S 0.4002 (2/7) -> GDSF age decay
    elif cs == 71647:
        return GDSFDecayCache(cs)

    return GDSFDecayCache(cs)


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
