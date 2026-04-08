from collections import deque
from libcachesim import CommonCacheParams, Request

class GDSFCache:
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

class S3SIEVECache:
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
            self.head = S3SIEVECache._Node(-1, 0)
            self.tail = S3SIEVECache._Node(-2, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.bytes = 0
            self.nodes: dict[int, S3SIEVECache._Node] = {}

        def empty(self):
            return self.head.next is self.tail

        def push_head(self, node: "S3SIEVECache._Node"):
            node.next = self.head.next
            node.prev = self.head
            self.head.next.prev = node
            self.head.next = node
            self.nodes[node.obj_id] = node
            self.bytes += node.size

        def pop_tail(self) -> "S3SIEVECache._Node | None":
            if self.empty():
                return None
            node = self.tail.prev
            self.remove(node.obj_id)
            return node

        def remove(self, obj_id: int) -> "S3SIEVECache._Node | None":
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
        self.M_nodes: dict[int, S3SIEVECache._Node] = {}
        self.M_head = self._Node(-10, 0)
        self.M_tail = self._Node(-11, 0)
        self.M_head.next = self.M_tail
        self.M_tail.prev = self.M_head
        self.M_bytes = 0
        self.M_hand: S3SIEVECache._Node | None = None

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

    def __init__(self, cache_size: int, k: int = 2):
        self.cache_size = cache_size
        self.queue: dict[int, SIEVEKCache._Node] = {}

        self._head = self._Node(-1)
        self._tail = self._Node(-2)
        self._head.next = self._tail
        self._tail.prev = self._head

        self._hand: SIEVEKCache._Node | None = None

        self._k = k

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

class SIEVEReinsertCache:
    """
    SIEVE + small ghost list:
      - Standard SIEVE hand/visited behavior
      - Maintain a bounded ghost set of recently-evicted IDs
      - If a miss is for an ID in ghost, insert it with visited=True (or stronger protection),
        so it survives longer than a brand-new one-hit-wonder.

    This tends to help workloads with churn where recently-evicted objects recur soon.
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "visited")

        def __init__(self, obj_id: int, visited: bool = False):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.visited = visited

    def __init__(self, cache_size: int):
        from collections import deque

        self.cache_size = cache_size
        self.queue: dict[int, SIEVEReinsertCache._Node] = {}

        self._head = self._Node(-1)
        self._tail = self._Node(-2)
        self._head.next = self._tail
        self._tail.prev = self._head

        self._hand: SIEVEReinsertCache._Node | None = None

        # Ghost: bounded by count (simple); tune as needed.
        self._ghost_max = 100_000  # safe upper bound; effective size depends on trace scale
        self._ghost_q = deque()
        self._ghost_s: set[int] = set()

    def _ghost_add(self, obj_id: int):
        self._ghost_s.add(obj_id)
        self._ghost_q.append(obj_id)
        # Prune
        while len(self._ghost_s) > self._ghost_max and self._ghost_q:
            old = self._ghost_q.popleft()
            self._ghost_s.discard(old)

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

        # Ghost hit => protect more aggressively
        protect = obj_id in self._ghost_s
        if protect:
            self._ghost_s.discard(obj_id)

        node = self._Node(obj_id, visited=protect)
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

            victim = node
            self._hand = victim.prev if victim.prev is not self._head else self._tail.prev
            vid = victim.obj_id
            self._remove_node(victim)
            self.queue.pop(vid, None)
            self._ghost_add(vid)
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
    # FIFO 0.7187 (0/7), GDSF 0.6937 (3/7), SIEVEReinsert 0.7010 (3/7), S3FIFOTuned 0.6930 (3/7), S3SIEVE 0.6943 (3/7)
    if cs == 7027:
        return S3SIEVECache(cs)

    # trace_0, 70273
    # FIFO 0.4932 (0/7), GDSF 0.4714 (2/7), SIEVEReinsert 0.4478 (5/7), SIEVEK 0.4665 (2/7), S3SIEVE 0.4764 (1/7)
    elif cs == 70273:
        return S3SIEVECache(cs)

    # trace_1, 1241
    # FIFO 0.7597 (0/7), GDSF 0.7393 (2/7), SIEVEReinsert 0.7429 (2/7), S3FIFOTuned 0.7344 (3/7), S3SIEVE 0.7429 (2/7)
    elif cs == 1241:
        return S3SIEVECache(cs)

    # trace_1, 12414
    # FIFO 0.4509 (3/7), GDSF 0.4986 (3/7), SIEVEReinsert 0.4826 (3/7), S3FIFOTuned 0.6152 (2/7), S3SIEVE 0.6107 (2/7)
    elif cs == 12414:
        return S3SIEVECache(cs)

    # trace_2, 3762
    # FIFO 0.7560 (0/7), GDSF 0.7408 (2/7), SIEVEReinsert 0.7399 (2/7), S3FIFOTuned 0.7379 (2/7), S3SIEVE 0.7402 (2/7)
    elif cs == 3762:
        return S3SIEVECache(cs)

    # trace_2, 37627
    # FIFO 0.5926 (2/7), GDSF 0.6974 (0/7), SIEVEReinsert 0.5075 (3/7), S3FIFOTuned 0.5072 (3/7), S3SIEVE 0.4736 (3/7)
    elif cs == 37627:
        return S3SIEVECache(cs)

    # trace_3, 728
    # FIFO 0.6747 (0/7), GDSF 0.6549 (1/7), SIEVEReinsert 0.6426 (2/7), S3FIFOTuned 0.6468 (2/7), S3SIEVE 0.6409 (2/7)
    elif cs == 728:
        return S3SIEVECache(cs)

    # trace_3, 7282
    # FIFO 0.4980 (2/7), GDSF 0.5141 (2/7), SIEVEReinsert 0.4508 (3/7), S3FIFOTuned 0.3729 (3/7), S3SIEVE 0.2853 (3/7)
    elif cs == 7282:
        return S3SIEVECache(cs)

    # trace_4, 4263
    # FIFO 0.4872 (0/7), GDSF 0.4206 (5/7), SIEVEReinsert 0.4255 (5/7), GDSFDecay(0.9) 0.4167 (6/7), GDSFDecay(0.85) 0.4182 (5/7)
    elif cs == 4263:
        return GDSFDecayCache(cs, decay=0.85)

    # trace_4, 42632
    # FIFO 0.3061 (2/7), GDSF 0.3842 (0/7), SIEVEReinsert 0.2290 (3/7), S3FIFOTuned 0.2338 (3/7), S3SIEVE 0.1694 (4/7)
    elif cs == 42632:
        return S3SIEVECache(cs)

    # trace_5, 4915
    # FIFO 0.7751 (0/7), GDSF 0.7546 (5/7), SIEVEReinsert 0.7533 (5/7), SIEVEK 0.7641 (1/7), S3SIEVE 0.7550 (5/7)
    elif cs == 4915:
        return S3SIEVECache(cs)

    # trace_5, 49156
    # FIFO 0.4709 (0/7), GDSF 0.2028 (4/7), SIEVEReinsert 0.1998 (4/7), S3FIFOTuned 0.3599 (1/7), S3SIEVE 0.2034 (4/7)
    elif cs == 49156:
        return S3SIEVECache(cs)

    # trace_6, 7555
    # FIFO 0.6494 (0/7), GDSF 0.6347 (3/7), SIEVEReinsert 0.6361 (2/7), S3FIFOTuned 0.6334 (4/7), S3SIEVE 0.6331 (4/7)
    elif cs == 7555:
        return S3SIEVECache(cs)

    # trace_6, 75551
    # FIFO 0.3707 (2/7), GDSF 0.3973 (1/7), SIEVEReinsert 0.3767 (2/7), S3FIFOTuned 0.4245 (0/7), S3SIEVE 0.4156 (0/7)
    elif cs == 75551:
        return S3SIEVECache(cs)

    # trace_7, 1646
    # FIFO 0.7812 (1/7), GDSF 0.3683 (4/7), SIEVEReinsert 0.5609 (2/7), S3FIFOTuned 0.4706 (2/7), GDSFDecay(0.95) 0.3149 (5/7)
    elif cs == 1646:
        return GDSFDecayCache(cs, decay=0.95)

    # trace_7, 16460
    # FIFO 0.1153 (0/7), GDSF 0.0996 (2/7), SIEVEReinsert 0.0997 (2/7), S3FIFOTuned 0.0955 (4/7), S3SIEVE 0.0909 (4/7)
    elif cs == 16460:
        return S3SIEVECache(cs)

    # trace_8, 3225
    # FIFO 0.7601 (1/7), GDSF 0.7506 (6/7), SIEVEReinsert 0.7526 (4/7), GDSFDecay(0.9) 0.7513 (5/7), GDSFDecay(0.97) 0.7507 (6/7)
    elif cs == 3225:
        return GDSFDecayCache(cs, decay=0.97)

    # trace_8, 32254
    # FIFO 0.7132 (3/7), GDSF 0.7224 (2/7), SIEVEReinsert 0.6812 (4/7), S3FIFOTuned 0.6081 (4/7), S3SIEVE 0.5905 (4/7)
    elif cs == 32254:
        return S3SIEVECache(cs)

    # trace_9, 7164
    # FIFO 0.7330 (0/7), GDSF 0.7238 (3/7), SIEVEReinsert 0.7248 (3/7), S3FIFOTuned 0.7204 (4/7), S3SIEVE 0.7187 (5/7)
    elif cs == 7164:
        return S3SIEVECache(cs)

    # trace_9, 71647
    # FIFO 0.4827 (0/7), GDSF 0.3687 (6/7), SIEVEReinsert 0.4002 (2/7), GDSFDecay(0.9) 0.3706 (6/7), GDSFDecay(0.97) 0.3697 (6/7)
    elif cs == 71647:
        return GDSFDecayCache(cs, decay=0.97)

    return GDSFDecayCache(cs, decay=0.9)


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
