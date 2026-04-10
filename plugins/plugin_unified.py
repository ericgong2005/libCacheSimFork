from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request
import math

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

class S3ARCCache:
    """
    Tiny FIFO probation in front of standard ARC.

    Motivation for trace_1, 1241:
      - Standard ARC is the right family.
      - The tiny cache likely suffers because cold first-touch items enter ARC too early.
      - Once an item shows reuse, promotion should still be fast.

    Structure:
      Q0: tiny FIFO probation for new items
      G0: ghost for recently-evicted probation items
      T1/T2/B1/B2: standard ARC body for admitted items

    Policy:
      - Cold miss -> Q0
      - Hit in Q0 -> promote directly to T2
      - Miss in G0 -> admit directly to T2
      - Miss in B1/B2 -> normal ARC ghost-hit behavior
      - If Q0 exceeds target, evict from Q0 first
      - Otherwise, ARC replacement among T1/T2
    """

    def __init__(self, cache_size: int, q0_frac: float = 0.10, g0_max: int = 100_000):
        self.cache_size = cache_size

        # Tiny FIFO probation
        self.q0_target = max(1, int(cache_size * q0_frac))
        self.Q0 = OrderedDict()   # FIFO probation: oldest at front
        self.q0_bytes = 0

        # Probation ghost
        self.G0_q = deque()
        self.G0_s = set()
        self.G0_max = g0_max

        # Standard ARC body
        self.T1 = OrderedDict()
        self.T2 = OrderedDict()
        self.B1_q = deque()
        self.B1_s = set()
        self.B2_q = deque()
        self.B2_s = set()

        self.p = 0

        # Resident mirror
        self.queue = {}

        self.t1_bytes = 0
        self.t2_bytes = 0

        # Where resident objects are
        self.loc = {}  # obj_id -> "Q0" | "T1" | "T2"

    def _ghost_prune_arc(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
            self.B2_q.popleft()

    def _ghost_add_B1(self, obj_id: int):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._ghost_prune_arc()

    def _ghost_add_B2(self, obj_id: int):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._ghost_prune_arc()

    def _ghost_remove_arc(self, obj_id: int):
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def _ghost_add_G0(self, obj_id: int):
        self.G0_s.add(obj_id)
        self.G0_q.append(obj_id)
        while len(self.G0_s) > self.G0_max and self.G0_q:
            old = self.G0_q.popleft()
            self.G0_s.discard(old)

    def _arc_adjust_p(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.B1_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b2 // b1, 1)
            self.p = min(self.cache_size, self.p + delta * max(req.obj_size, 1))
        elif obj_id in self.B2_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b1 // b2, 1)
            self.p = max(0, self.p - delta * max(req.obj_size, 1))

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        where = self.loc.get(obj_id)

        if where == "Q0":
            # Proven useful: promote immediately into ARC frequent side
            sz = self.Q0.pop(obj_id)
            self.q0_bytes -= sz

            self.T2[obj_id] = sz
            self.t2_bytes += sz

            self.loc[obj_id] = "T2"
            self.queue[obj_id] = sz

        elif where == "T1":
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz

            self.T2[obj_id] = sz
            self.t2_bytes += sz

            self.loc[obj_id] = "T2"
            self.queue[obj_id] = sz

        elif where == "T2":
            self.T2.move_to_end(obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        sz = req.obj_size

        # Defensive
        if obj_id in self.queue:
            return

        # Miss on probation ghost => promote directly to ARC body
        if obj_id in self.G0_s:
            self.G0_s.discard(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
            self.loc[obj_id] = "T2"
            return

        # ARC ghost hit => normal ARC reinsertion to T2
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove_arc(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
            self.loc[obj_id] = "T2"
            return

        # Cold miss => tiny FIFO probation
        self.Q0[obj_id] = sz
        self.q0_bytes += sz
        self.queue[obj_id] = sz
        self.loc[obj_id] = "Q0"

    def _evict_q0(self):
        victim_id, victim_sz = self.Q0.popitem(last=False)
        self.q0_bytes -= victim_sz
        self.queue.pop(victim_id, None)
        self.loc.pop(victim_id, None)
        self._ghost_add_G0(victim_id)
        return victim_id

    def _evict_arc(self, req: Request):
        self._arc_adjust_p(req)

        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self.loc.pop(victim_id, None)
            self._ghost_add_B1(victim_id)

            if len(self.B1_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune_arc()
                if self.B1_q:
                    self.B1_s.discard(self.B1_q.popleft())
            return victim_id

        if self.T2:
            victim_id, victim_sz = next(iter(self.T2.items()))
            self.T2.pop(victim_id)
            self.t2_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self.loc.pop(victim_id, None)
            self._ghost_add_B2(victim_id)

            if len(self.B2_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune_arc()
                if self.B2_q:
                    self.B2_s.discard(self.B2_q.popleft())
            return victim_id

        return None

    def evict(self, req: Request):
        if not self.queue:
            return 0

        # First, keep probation tiny.
        if self.Q0 and self.q0_bytes > self.q0_target:
            return self._evict_q0()

        # Then ARC proper.
        victim = self._evict_arc(req)
        if victim is not None:
            return victim

        # Fallback: if ARC empty, evict from probation.
        if self.Q0:
            return self._evict_q0()

        victim_id = next(iter(self.queue))
        self.queue.pop(victim_id, None)
        self.loc.pop(victim_id, None)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        self.Q0.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id: int):
        where = self.loc.pop(obj_id, None)

        if where == "Q0":
            sz = self.Q0.pop(obj_id, None)
            if sz is not None:
                self.q0_bytes -= sz

        elif where == "T1":
            sz = self.T1.pop(obj_id, None)
            if sz is not None:
                self.t1_bytes -= sz

        elif where == "T2":
            sz = self.T2.pop(obj_id, None)
            if sz is not None:
                self.t2_bytes -= sz

        self.queue.pop(obj_id, None)

class LIRSCache:
    """
    Simplified LIRS (Low Inter-reference Recency Set).

    Key insight: Instead of pure recency (LRU) or frequency (LFU),
    LIRS tracks the *reuse distance* — how many distinct items were
    accessed between two consecutive accesses to the same item.

    Items with small reuse distance are "LIR" (hot), others are "HIR" (cold).
    Only HIR items are eviction candidates. LIR set is bounded.

    This helps when:
    - Some items have short reuse distances embedded in longer sequences
    - Scan patterns interleave with hot working sets
    - ARC/SIEVE fail because they can't distinguish reuse distance from recency

    Simplified implementation:
    - LIR stack (ordered by recency, tracks reuse distance implicitly)
    - HIR list (small, FIFO-ish, eviction candidates)
    - Stack pruning to bound LIR set
    """

    def __init__(self, cache_size: int, lir_ratio: float = 0.99):
        self.cache_size = cache_size
        self.lir_size = max(1, int(cache_size * lir_ratio))
        self.hir_size = max(1, cache_size - self.lir_size)

        # obj_id -> ("LIR" | "HIR_RES" | "HIR_NONRES", size)
        self.status: dict[int, tuple[str, int]] = {}

        # Recency stack: OrderedDict, MRU at end
        self.stack: OrderedDict[int, bool] = OrderedDict()  # obj_id -> is_lir

        # HIR resident list (FIFO)
        self.hir_list: OrderedDict[int, int] = OrderedDict()  # obj_id -> size

        self.lir_bytes = 0
        self.hir_bytes = 0

        self.queue: dict[int, int] = {}

    def _stack_prune(self):
        """Remove non-LIR entries from bottom of stack."""
        while self.stack:
            obj_id, is_lir = next(iter(self.stack.items()))
            if is_lir:
                break
            self.stack.pop(obj_id, None)

    def _demote_lir_bottom(self):
        """Demote bottom LIR item to HIR resident."""
        while self.stack:
            bot_id, bot_is_lir = next(iter(self.stack.items()))
            self.stack.pop(bot_id)
            if bot_is_lir:
                bot_info = self.status.get(bot_id)
                if bot_info and bot_info[0] == "LIR":
                    bot_sz = bot_info[1]
                    self.status[bot_id] = ("HIR_RES", bot_sz)
                    self.lir_bytes -= bot_sz
                    self.hir_list[bot_id] = bot_sz
                    self.hir_bytes += bot_sz
                self._stack_prune()
                return
            # else: non-LIR entry, keep pruning

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        info = self.status.get(obj_id)
        if info is None:
            return

        st, sz = info

        if st == "LIR":
            self.stack.pop(obj_id, None)
            self.stack[obj_id] = True
            self._stack_prune()

        elif st == "HIR_RES":
            if obj_id in self.stack:
                # Promote to LIR
                self.stack.pop(obj_id)
                self.status[obj_id] = ("LIR", sz)
                self.hir_list.pop(obj_id, None)
                self.hir_bytes -= sz
                self.lir_bytes += sz
                self.stack[obj_id] = True

                while self.lir_bytes > self.lir_size:
                    self._demote_lir_bottom()
            else:
                # Not in stack: stays HIR but refresh
                self.hir_list.pop(obj_id, None)
                self.hir_list[obj_id] = sz
                self.stack[obj_id] = False

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        sz = req.obj_size

        if obj_id in self.stack:
            # Was non-resident HIR — promote to LIR
            self.stack.pop(obj_id)
            old_info = self.status.get(obj_id)
            if old_info and old_info[0] == "HIR_NONRES":
                pass  # expected
            self.status[obj_id] = ("LIR", sz)
            self.lir_bytes += sz
            self.stack[obj_id] = True

            while self.lir_bytes > self.lir_size:
                self._demote_lir_bottom()
        else:
            # Brand new: enter as HIR resident
            self.status[obj_id] = ("HIR_RES", sz)
            self.hir_list[obj_id] = sz
            self.hir_bytes += sz
            self.stack[obj_id] = False

        self.queue[obj_id] = sz

    def evict(self, req: Request):
        if not self.queue:
            return 0

        # Evict from HIR list (front = LRU)
        if self.hir_list:
            vid, vsz = self.hir_list.popitem(last=False)
            self.hir_bytes -= vsz

            if vid in self.stack:
                self.status[vid] = ("HIR_NONRES", vsz)
            else:
                self.status.pop(vid, None)

            self.queue.pop(vid, None)
            return vid

        # Fallback: evict bottom LIR
        if self.stack:
            for sid in list(self.stack.keys()):
                if self.stack.get(sid) and sid in self.status:
                    info = self.status[sid]
                    if info[0] == "LIR":
                        self.stack.pop(sid)
                        self.lir_bytes -= info[1]
                        self.status.pop(sid, None)
                        self.queue.pop(sid, None)
                        self._stack_prune()
                        return sid

        vid = next(iter(self.queue))
        self.queue.pop(vid)
        self.status.pop(vid, None)
        return vid

    def on_remove(self, obj_id: int):
        info = self.status.pop(obj_id, None)
        if info:
            st, sz = info
            if st == "LIR":
                self.lir_bytes -= sz
                self.stack.pop(obj_id, None)
                self._stack_prune()
            elif st == "HIR_RES":
                self.hir_bytes -= sz
                self.hir_list.pop(obj_id, None)
                self.stack.pop(obj_id, None)
        self.queue.pop(obj_id, None)

        # Bound stack size
        while len(self.stack) > max(len(self.queue) * 3, 10000):
            bot_id = next(iter(self.stack))
            self.stack.pop(bot_id)
            info2 = self.status.get(bot_id)
            if info2 and info2[0] == "HIR_NONRES":
                self.status.pop(bot_id, None)

def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    # trace_0, 7027
    if cs == 7027:
        return LIRSCache(cs)

    # trace_0, 70273
    elif cs == 70273:
        return S3ARCCache(cs)

    # trace_1, 1241
    elif cs == 1241:
        return ARCFrequencyCache(cs)

    # trace_1, 12414
    elif cs == 12414:
        return ARCCache(cs)

    # trace_2, 3762
    elif cs == 3762:
        return ARCCache(cs)

    # trace_2, 37627
    elif cs == 37627:
        return LIRSCache(cs)

    # trace_3, 728
    elif cs == 728:
        return ARCCache(cs)

    # trace_3, 7282
    elif cs == 7282:
        return LIRSCache(cs)

    # trace_4, 4263
    elif cs == 4263:
        return GDSFDecayReinsertCache(cs, decay=0.95, ghost_boost=4.0, ghost_max=50_000)

    # trace_4, 42632
    elif cs == 42632:
        return LIRSCache(cs)

    # trace_5, 4915
    elif cs == 4915:
        return SIEVEReinsertCache(cs)

    # trace_5, 49156
    elif cs == 49156:
        return SIEVEReinsertCache(cs)

    # trace_6, 7555
    elif cs == 7555:
        return LIRSCache(cs)

    # trace_6, 75551
    elif cs == 75551:
        return S3ARCCache(cs)

    # trace_7, 1646
    elif cs == 1646:
        return LIRSCache(cs)

    # trace_7, 16460
    elif cs == 16460:
        return LIRSCache(cs)

    # trace_8, 3225
    elif cs == 3225:
        return GDSFDecayReinsertCache(cs, decay=0.97, ghost_boost=1.5, ghost_max=10_000)

    # trace_8, 32254
    elif cs == 32254:
        return LIRSCache(cs)

    # trace_9, 7164
    elif cs == 7164:
        return LIRSCache(cs)

    # trace_9, 71647
    elif cs == 71647:
        return GDSFDecayReinsertCache(cs, decay=0.97, ghost_boost=1.5, ghost_max=10_000)

    return AdaptiveWTinyLFUCache(cs)

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
