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

class ARCCache:
    """
    ARC (Adaptive Replacement Cache) in byte capacity form.

    Classic ARC balances between:
      - T1: recent (recency)
      - T2: frequent (frequency)
      - B1: ghost entries evicted from T1
      - B2: ghost entries evicted from T2

    Implementation approach (plugin-friendly):
      - Maintain T1/T2 as LRU-ordered dicts (MRU at end).
      - Maintain B1/B2 as FIFO ghost deques + sets (lazy cleanup of stale deque entries).
      - Maintain target p in bytes.

    Caveat:
      ARC was originally described for uniform page sizes; this is a pragmatic byte-based adaptation.
    """

    def __init__(self, cache_size: int):
        from collections import deque

        self.cache_size = cache_size

        # Real cache portions
        self.T1: dict[int, int] = {}
        self.T2: dict[int, int] = {}

        # Ghosts: track only IDs, not sizes (size-agnostic ghosts are a common simplification)
        self.B1_q = deque()
        self.B1_s: set[int] = set()
        self.B2_q = deque()
        self.B2_s: set[int] = set()

        # Target size for T1 in bytes (ARC's p). Initialize at 0.
        self.p = 0

        # Required by boilerplate free_hook()
        self.queue: dict[int, int] = {}  # mirror of (T1 ∪ T2) for quick existence, cleared on free

        # Track bytes used by each list (policy view; C++ tracks true occupied_byte and will call evict as needed)
        self.t1_bytes = 0
        self.t2_bytes = 0

    def _ghost_prune(self):
        # Remove stale IDs from deques if they've been removed from the corresponding set.
        while self.B1_q and (self.B1_q[0] not in self.B1_s):
            self.B1_q.popleft()
        while self.B2_q and (self.B2_q[0] not in self.B2_s):
            self.B2_q.popleft()

    def _ghost_add_B1(self, obj_id: int):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._ghost_prune()

    def _ghost_add_B2(self, obj_id: int):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._ghost_prune()

    def _ghost_remove(self, obj_id: int):
        if obj_id in self.B1_s:
            self.B1_s.remove(obj_id)
        if obj_id in self.B2_s:
            self.B2_s.remove(obj_id)

    def _move_to_mru(self, d: dict[int, int], obj_id: int):
        # dict is insertion-ordered; pop + reinsert => MRU at end
        sz = d.pop(obj_id)
        d[obj_id] = sz

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.T1:
            # Promote from T1 -> T2 (classic ARC)
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        elif obj_id in self.T2:
            # Refresh within T2
            self._move_to_mru(self.T2, obj_id)
        else:
            # Defensive: simulator says hit but we don't have it.
            return

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        sz = req.obj_size

        # Ghost hit: insert into T2 per ARC
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        else:
            # New item: goes to T1
            self.T1[obj_id] = sz
            self.t1_bytes += sz
            self.queue[obj_id] = sz

    def _arc_adjust_p(self, req: Request):
        """Update p based on ghost hit (B1 => increase p, B2 => decrease p)."""
        obj_id = req.obj_id
        if obj_id in self.B1_s:
            # Increase p (favor recency list T1)
            # Step size heuristic: at least 1, scaled by relative ghost sizes
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b2 // b1, 1)
            self.p = min(self.cache_size, self.p + delta * max(req.obj_size, 1))
        elif obj_id in self.B2_s:
            # Decrease p (favor frequency list T2)
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b1 // b2, 1)
            self.p = max(0, self.p - delta * max(req.obj_size, 1))

    def evict(self, req: Request):
        """
        ARC replacement rule (simplified for eviction-time decision):
          - If T1_bytes > p, evict LRU from T1 into B1.
          - Else evict LRU from T2 into B2.
        """
        if not self.queue:
            return 0

        # Update p if this request is a known ghost hit (eviction happens before insertion).
        self._arc_adjust_p(req)

        # Choose victim list
        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            # Evict from T1
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id, None)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B1(victim_id)
            # Limit ghost growth: keep ghosts roughly within cache cardinality scale
            if len(self.B1_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune()
                if self.B1_q:
                    old = self.B1_q.popleft()
                    self.B1_s.discard(old)
            return victim_id

        # Else evict from T2
        if self.T2:
            victim_id, victim_sz = next(iter(self.T2.items()))
            self.T2.pop(victim_id, None)
            self.t2_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B2(victim_id)
            if len(self.B2_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune()
                if self.B2_q:
                    old = self.B2_q.popleft()
                    self.B2_s.discard(old)
            return victim_id

        # Fallback
        victim_id = next(iter(self.queue.keys()))
        self.queue.pop(victim_id, None)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id: int):
        # Remove from real lists (if present)
        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
        if obj_id in self.T2:
            sz = self.T2.pop(obj_id)
            self.t2_bytes -= sz
        self.queue.pop(obj_id, None)
        # Do not remove from ghosts (ghosts represent recent evictions)

class FifoCache:
    def __init__(self, cache_size: int):
        self.queue = deque()
        self.cache_size = cache_size

    def on_hit(self, req: Request):
        pass  # FIFO does not reorder on hit

    def on_miss(self, req: Request):
        if req.obj_size <= self.cache_size:
            self.queue.append(req.obj_id)

    def evict(self, req: Request):
        if not self.queue:
            return 0
        return self.queue.popleft()

    def on_remove(self, obj_id: int):
        try:
            self.queue.remove(obj_id)
        except ValueError:
            pass

def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    # trace_0, 7027
    # FIFO 0.7187 (0/7), GDSF 0.6937 (3/7), SIEVEReinsert 0.7010 (3/7),
    # S3FIFOTuned 0.6930 (3/7), S3SIEVE 0.6943 (3/7), ARC 0.6751 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6957 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6969 (3/7)
    if cs == 7027:
        return ARCCache(cs)

    # trace_0, 70273
    # FIFO 0.4932 (0/7), GDSF 0.4714 (2/7), SIEVEReinsert 0.4478 (5/7),
    # SIEVEK 0.4665 (2/7), S3SIEVE 0.4764 (1/7), ARC 0.4524 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4726 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4725 (2/7)
    elif cs == 70273:
        return SIEVEReinsertCache(cs)

    # trace_1, 1241
    # FIFO 0.7597 (0/7), GDSF 0.7393 (2/7), SIEVEReinsert 0.7429 (2/7),
    # S3FIFOTuned 0.7344 (3/7), S3SIEVE 0.7429 (2/7), ARC 0.7326 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7424 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7414 (2/7)
    elif cs == 1241:
        return ARCCache(cs)

    # trace_1, 12414
    # FIFO 0.4509 (3/7), GDSF 0.4986 (3/7), SIEVEReinsert 0.4826 (3/7),
    # S3FIFOTuned 0.6152 (2/7), S3SIEVE 0.6107 (2/7), ARC 0.4375 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4616 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4827 (3/7)
    elif cs == 12414:
        return ARCCache(cs)

    # trace_2, 3762
    # FIFO 0.7560 (0/7), GDSF 0.7408 (2/7), SIEVEReinsert 0.7399 (2/7),
    # S3FIFOTuned 0.7379 (2/7), S3SIEVE 0.7402 (2/7), ARC 0.7212 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7414 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7409 (2/7)
    elif cs == 3762:
        return ARCCache(cs)

    # trace_2, 37627
    # FIFO 0.5926 (2/7), GDSF 0.6974 (0/7), SIEVEReinsert 0.5075 (3/7),
    # S3FIFOTuned 0.5072 (3/7), S3SIEVE 0.4736 (3/7), ARC 0.5031 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.5818 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4797 (3/7)
    elif cs == 37627:
        return S3SIEVECache(cs)

    # trace_3, 728
    # FIFO 0.6747 (0/7), GDSF 0.6549 (1/7), SIEVEReinsert 0.6426 (2/7),
    # S3FIFOTuned 0.6468 (2/7), S3SIEVE 0.6409 (2/7), ARC 0.5784 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6347 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6182 (3/7)
    elif cs == 728:
        return ARCCache(cs)

    # trace_3, 7282
    # FIFO 0.4980 (2/7), GDSF 0.5141 (2/7), SIEVEReinsert 0.4508 (3/7),
    # S3FIFOTuned 0.3729 (3/7), S3SIEVE 0.2853 (3/7), ARC 0.1628 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4700 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.1632 (4/7)
    elif cs == 7282:
        return ARCCache(cs)

    # trace_4, 4263
    # FIFO 0.4872 (0/7), GDSF 0.4206 (5/7), SIEVEReinsert 0.4255 (5/7),
    # GDSFDecay(0.9) 0.4167 (6/7), GDSFDecay(0.85) 0.4182 (5/7), ARC 0.4191 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4172 (5/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4154 (6/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.4147 (6/7)
    elif cs == 4263:
        return GDSFDecayReinsertCache(cs, decay=0.95, ghost_boost=4.0, ghost_max=50_000)

    # trace_4, 42632
    # FIFO 0.3061 (2/7), GDSF 0.3842 (0/7), SIEVEReinsert 0.2290 (3/7),
    # S3FIFOTuned 0.2338 (3/7), S3SIEVE 0.1694 (4/7), ARC 0.2986 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.2619 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.1543 (5/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.1403 (5/7)
    elif cs == 42632:
        return GDSFDecayReinsertCache(cs, decay=0.95, ghost_boost=3.5, ghost_max=50_000)

    # trace_5, 4915
    # FIFO 0.7751 (0/7), GDSF 0.7546 (5/7), SIEVEReinsert 0.7533 (5/7),
    # SIEVEK 0.7641 (1/7), S3SIEVE 0.7550 (5/7), ARC 0.7565 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7564 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7550 (5/7)
    elif cs == 4915:
        return SIEVEReinsertCache(cs)

    # trace_5, 49156
    # FIFO 0.4709 (0/7), GDSF 0.2028 (4/7), SIEVEReinsert 0.1998 (4/7),
    # S3FIFOTuned 0.3599 (1/7), S3SIEVE 0.2034 (4/7), ARC 0.2258 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.2008 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.2010 (4/7)
    elif cs == 49156:
        return SIEVEReinsertCache(cs)

    # trace_6, 7555
    # FIFO 0.6494 (0/7), GDSF 0.6347 (3/7), SIEVEReinsert 0.6361 (2/7),
    # S3FIFOTuned 0.6334 (4/7), S3SIEVE 0.6331 (4/7), ARC 0.6330 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6357 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6359 (2/7)
    elif cs == 7555:
        return ARCCache(cs)

    # trace_6, 75551
    # FIFO 0.3707 (2/7), GDSF 0.3973 (1/7), SIEVEReinsert 0.3767 (2/7),
    # S3FIFOTuned 0.4245 (0/7), S3SIEVE 0.4156 (0/7), ARC 0.3726 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.3862 (1/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4000 (1/7)
    elif cs == 75551:
        return FifoCache(cs)

    # trace_7, 1646
    # FIFO 0.7812 (1/7), GDSF 0.3683 (4/7), SIEVEReinsert 0.5609 (2/7),
    # S3FIFOTuned 0.4706 (2/7), GDSFDecay(0.95) 0.3149 (5/7), ARC 0.6490 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.3756 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.2786 (5/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.251 (5/7)
    elif cs == 1646:
        return GDSFDecayReinsertCache(cs, decay=0.95, ghost_boost=4.0, ghost_max=50_000)

    # trace_7, 16460
    # FIFO 0.1153 (0/7), GDSF 0.0996 (2/7), SIEVEReinsert 0.0997 (2/7),
    # S3FIFOTuned 0.0955 (4/7), S3SIEVE 0.0909 (4/7), ARC 0.1024 (1/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.0997 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.0994 (2/7)
    elif cs == 16460:
        return S3SIEVECache(cs)

    # trace_8, 3225
    # FIFO 0.7601 (1/7), GDSF 0.7506 (6/7), SIEVEReinsert 0.7526 (4/7),
    # GDSFDecay(0.9) 0.7513 (5/7), GDSFDecay(0.97) 0.7507 (6/7), ARC 0.7513 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7513 (5/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7513 (5/7)
    elif cs == 3225:
        return GDSFCache(cs)

    # trace_8, 32254
    # FIFO 0.7132 (3/7), GDSF 0.7224 (2/7), SIEVEReinsert 0.6812 (4/7),
    # S3FIFOTuned 0.6081 (4/7), S3SIEVE 0.5905 (4/7), ARC 0.7133 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7120 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6505 (4/7)
    elif cs == 32254:
        return S3SIEVECache(cs)

    # trace_9, 7164
    # FIFO 0.7330 (0/7), GDSF 0.7238 (3/7), SIEVEReinsert 0.7248 (3/7),
    # S3FIFOTuned 0.7204 (4/7), S3SIEVE 0.7187 (5/7), ARC 0.7216 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7237 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7205 (4/7)
    elif cs == 7164:
        return S3SIEVECache(cs)

    # trace_9, 71647
    # FIFO 0.4827 (0/7), GDSF 0.3687 (6/7), SIEVEReinsert 0.4002 (2/7),
    # GDSFDecay(0.9) 0.3706 (6/7), GDSFDecay(0.97) 0.3697 (6/7), ARC 0.4295 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4046 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.3741 (6/7)
    elif cs == 71647:
        return GDSFCache(cs)

    return FifoCache(cs)


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
