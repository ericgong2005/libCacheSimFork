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

class LECARCache:
    """
    LeCaR with improved discount handling.

    Fix: original used cumulative time for discount, causing
    (1-d)^t -> 0 after a few hundred misses, freezing weights.
    Now uses time-since-eviction per ghost entry, keeping
    learning active throughout the trace.

    Also supports swapping LRU for FIFO as one policy arm,
    useful for scan-heavy workloads where FIFO > LRU.
    """

    def __init__(self, cache_size: int, learning_rate: float = 0.45,
                 discount: float = 0.005, use_fifo: bool = False):
        import heapq
        import random

        self.cache_size = cache_size
        self.lr = learning_rate
        self.discount = discount
        self.use_fifo = use_fifo
        self._heapq = heapq
        self._random = random

        self.w = 0.5

        # Recency policy: OrderedDict (LRU or FIFO depending on use_fifo)
        self.recency_order: OrderedDict[int, int] = OrderedDict()

        # LFU: dict + lazy heap
        self.freq: dict[int, int] = {}
        self.heap: list[tuple[int, int, int]] = []
        self._ver = 0

        # Ghost lists — now store eviction timestamp for proper discount
        self.ghost_rec: OrderedDict[int, int] = OrderedDict()  # obj_id -> eviction_time
        self.ghost_freq: OrderedDict[int, int] = OrderedDict()  # obj_id -> eviction_time
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
        self._time += 1
        # Update recency: move to MRU (LRU mode) or no-op (FIFO mode)
        if not self.use_fifo and obj_id in self.recency_order:
            self.recency_order.move_to_end(obj_id)
        # Update frequency
        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1
        self._lfu_push(obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        sz = req.obj_size
        self._time += 1

        # Learn from ghost hits using time-since-eviction for discount
        if obj_id in self.ghost_rec:
            evict_time = self.ghost_rec.pop(obj_id)
            age = self._time - evict_time
            d = math.pow(1 - self.discount, age)
            # Recency ghost hit -> recency was wrong -> favor frequency
            self.w = max(0.001, self.w * math.exp(-self.lr * d))

        elif obj_id in self.ghost_freq:
            evict_time = self.ghost_freq.pop(obj_id)
            age = self._time - evict_time
            d = math.pow(1 - self.discount, age)
            # Frequency ghost hit -> frequency was wrong -> favor recency
            self.w = min(0.999, 1.0 - (1.0 - self.w) * math.exp(-self.lr * d))

        self.queue[obj_id] = sz
        self.recency_order[obj_id] = sz
        self.freq[obj_id] = 1
        self._lfu_push(obj_id)

    def evict(self, req: Request):
        if not self.queue:
            return 0

        if self._random.random() < self.w:
            vid = self._evict_recency()
            if vid is not None:
                self.ghost_rec[vid] = self._time
                if len(self.ghost_rec) > self.ghost_max:
                    self.ghost_rec.popitem(last=False)
                return vid
            vid = self._evict_freq()
            if vid is not None:
                return vid
        else:
            vid = self._evict_freq()
            if vid is not None:
                self.ghost_freq[vid] = self._time
                if len(self.ghost_freq) > self.ghost_max:
                    self.ghost_freq.popitem(last=False)
                return vid
            vid = self._evict_recency()
            if vid is not None:
                return vid

        vid = next(iter(self.queue))
        self.queue.pop(vid)
        return vid

    def _evict_recency(self):
        while self.recency_order:
            vid, vsz = self.recency_order.popitem(last=False)
            if vid in self.queue:
                self.queue.pop(vid)
                self.freq.pop(vid, None)
                return vid
        return None

    def _evict_freq(self):
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
            self.recency_order.pop(obj_id, None)
            self.freq.pop(obj_id, None)
            return obj_id
        return None

    def on_remove(self, obj_id: int):
        self.queue.pop(obj_id, None)
        self.recency_order.pop(obj_id, None)
        self.freq.pop(obj_id, None)

class LECARByteAwareCache:
    """
    Deterministic, byte-aware LeCaR-style cache for variable-size objects.

    Design goals:
      - No randomness at all.
      - Byte-aware frequency protection.
      - Recency expert that can prefer either LRU-like or FIFO-like victims
        based on observed ghost feedback.
      - Heuristic deterministic victim selection from candidate objects.
      - Conservative admission for large first-seen objects to reduce scan pollution.

    Core idea:
      Maintain two learned balances:

        w_rf   : recency-vs-frequency preference
        w_lru  : LRU-vs-FIFO preference inside the recency expert

      But unlike stochastic LeCaR, these weights are used only as deterministic
      blend coefficients in candidate scoring.

    Objects tracked:
      - queue[obj_id] = size
      - recency_order: insertion/access order for LRU/FIFO candidate generation
      - freq[obj_id] = hit count
      - heap stores lazy-updated LFU-by-size priorities

    Ghosts:
      - ghost_rec[obj_id]  = (evict_time, size, evict_freq, rec_kind)
            rec_kind in {"lru", "fifo"}
      - ghost_freq[obj_id] = (evict_time, size, evict_freq)

      On a ghost hit:
        * ghost_rec  => recency expert made a mistake, shift toward frequency
        * ghost_freq => frequency expert made a mistake, shift toward recency

        If ghost_rec contains rec_kind:
          * "lru"  => shift recency subweight away from LRU toward FIFO
          * "fifo" => shift recency subweight away from FIFO toward LRU

    Deterministic eviction:
      1. Produce three candidates:
           - LRU victim
           - FIFO victim
           - FREQ victim (lowest freq/size utility)
      2. Score each candidate with a blended "badness" heuristic.
      3. Evict the candidate with the highest badness.

    Important note:
      This is not canonical LeCaR anymore; it is a deterministic adaptation
      for byte-sized caches and mixed IO traces.
    """

    def __init__(
        self,
        cache_size: int,
        learning_rate: float = 0.45,
        discount: float = 0.005,
        ghost_max: int = 100_000,
        admit_large_frac: float = 0.01,
        initial_w_rf: float = 0.50,
        initial_w_lru: float = 0.50,
        bypass_large_first_seen: bool = True,
    ):
        import heapq

        self.cache_size = cache_size
        self.lr = learning_rate
        self.discount = discount
        self.ghost_max = ghost_max
        self.admit_large_frac = admit_large_frac
        self.bypass_large_first_seen = bypass_large_first_seen

        self._heapq = heapq

        # Learned balances
        self.w_rf = float(min(0.999, max(0.001, initial_w_rf)))   # recency vs frequency
        self.w_lru = float(min(0.999, max(0.001, initial_w_lru))) # LRU vs FIFO

        # Resident objects
        self.queue: dict[int, int] = {}         # obj_id -> size
        self.freq: dict[int, int] = {}          # obj_id -> count
        self.size: dict[int, int] = {}          # obj_id -> size
        self.last_access: dict[int, int] = {}   # obj_id -> logical time of last touch
        self.insert_time: dict[int, int] = {}   # obj_id -> logical time of admission

        # Recency ordering:
        # insertion order retained for FIFO semantics
        # move_to_end on hit for LRU semantics
        self.recency_order: OrderedDict[int, int] = OrderedDict()

        # Lazy min-heap for low utility (freq/size) candidates
        # (utility, version, obj_id)
        self.heap: list[tuple[float, int, int]] = []
        self._ver = 0
        self._obj_ver: dict[int, int] = {}

        # Ghost metadata
        # ghost_rec[obj_id]  = (evict_time, size, evict_freq, rec_kind)
        # ghost_freq[obj_id] = (evict_time, size, evict_freq)
        self.ghost_rec: OrderedDict[int, tuple[int, int, int, str]] = OrderedDict()
        self.ghost_freq: OrderedDict[int, tuple[int, int, int]] = OrderedDict()

        self._time = 0

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _touch_time(self):
        self._time += 1
        return self._time

    def _min_size(self, sz: int) -> int:
        return max(int(sz), 1)

    def _utility(self, obj_id: int) -> float:
        """
        Frequency utility per byte.
        Higher utility => should be kept.
        Lower utility => better eviction candidate for frequency expert.
        """
        f = self.freq.get(obj_id, 0)
        sz = self._min_size(self.size.get(obj_id, 1))
        return f / sz

    def _push_freq_candidate(self, obj_id: int):
        self._ver += 1
        self._obj_ver[obj_id] = self._ver
        u = self._utility(obj_id)
        self._heapq.heappush(self.heap, (u, self._ver, obj_id))

    def _prune_ghosts(self):
        while len(self.ghost_rec) > self.ghost_max:
            self.ghost_rec.popitem(last=False)
        while len(self.ghost_freq) > self.ghost_max:
            self.ghost_freq.popitem(last=False)

    def _ghost_penalty(self, evict_time: int, obj_size: int) -> float:
        """
        Cost-weighted, age-discounted learning signal.

        Larger objects matter more.
        Faster re-reference matters more.
        """
        age = max(1, self._time - evict_time)
        # Exponential decay by age; bounded away from 0 by finite precision
        d = math.pow(max(1e-9, 1.0 - self.discount), age)

        size_factor = min(1.0, max(obj_size, 1) / max(self.cache_size, 1))
        # Keep penalty meaningful even for small objects
        size_factor = max(size_factor, 0.05)

        return d * size_factor

    def _update_w_rf_toward_frequency(self, penalty: float):
        # Recency was wrong -> reduce recency preference
        self.w_rf = max(0.001, self.w_rf * math.exp(-self.lr * penalty))

    def _update_w_rf_toward_recency(self, penalty: float):
        # Frequency was wrong -> increase recency preference
        self.w_rf = min(0.999, 1.0 - (1.0 - self.w_rf) * math.exp(-self.lr * penalty))

    def _update_w_lru_toward_fifo(self, penalty: float):
        # LRU specifically was wrong
        self.w_lru = max(0.001, self.w_lru * math.exp(-self.lr * penalty))

    def _update_w_lru_toward_lru(self, penalty: float):
        # FIFO specifically was wrong
        self.w_lru = min(0.999, 1.0 - (1.0 - self.w_lru) * math.exp(-self.lr * penalty))

    def _is_ghost_hit(self, obj_id: int) -> bool:
        return obj_id in self.ghost_rec or obj_id in self.ghost_freq

    def _should_admit(self, obj_id: int, sz: int) -> bool:
        """
        Deterministic admission control:
          - always admit ghost hits
          - always admit small objects
          - optionally bypass large first-seen objects
        """
        if obj_id in self.ghost_rec or obj_id in self.ghost_freq:
            return True

        threshold = max(1, int(self.cache_size * self.admit_large_frac))
        is_small = sz <= threshold

        if is_small:
            return True

        if not self.bypass_large_first_seen:
            return True

        return False

    def _candidate_lru(self):
        """
        LRU victim: oldest by access order.
        recency_order is access order because hits move_to_end().
        """
        for obj_id in self.recency_order.keys():
            if obj_id in self.queue:
                return obj_id
        return None

    def _candidate_fifo(self):
        """
        FIFO victim: oldest by insertion time.
        Since recency_order is mutated for LRU, use explicit insert_time.
        """
        best_id = None
        best_insert = None
        for obj_id in self.queue.keys():
            t = self.insert_time.get(obj_id, 0)
            if best_insert is None or t < best_insert:
                best_insert = t
                best_id = obj_id
        return best_id

    def _candidate_freq(self):
        """
        Lowest utility (freq/size) victim.
        Lazy heap with versions.
        """
        while self.heap:
            u, ver, obj_id = self.heap[0]
            if obj_id not in self.queue:
                self._heapq.heappop(self.heap)
                continue
            if self._obj_ver.get(obj_id) != ver:
                self._heapq.heappop(self.heap)
                continue
            cur_u = self._utility(obj_id)
            if abs(cur_u - u) > 1e-15:
                self._heapq.heappop(self.heap)
                continue
            return obj_id
        return None

    def _norm_recency_age(self, obj_id: int) -> float:
        """
        0..1 where larger means colder by access recency.
        """
        last = self.last_access.get(obj_id, self.insert_time.get(obj_id, self._time))
        age = max(0, self._time - last)
        denom = max(len(self.queue), 1)
        return min(1.0, age / denom)

    def _norm_fifo_age(self, obj_id: int) -> float:
        """
        0..1 where larger means older by insertion age.
        """
        ins = self.insert_time.get(obj_id, self._time)
        age = max(0, self._time - ins)
        denom = max(len(self.queue), 1)
        return min(1.0, age / denom)

    def _norm_low_utility(self, obj_id: int) -> float:
        """
        0..1 where larger means worse utility and more evictable.
        """
        # Map utility = f/size into an inverse badness.
        # Utility can be very small, so use u/(1+u) squashing then invert.
        u = self._utility(obj_id)
        keepiness = u / (1.0 + u)
        return 1.0 - keepiness

    def _norm_size_pressure(self, obj_id: int) -> float:
        """
        0..1 where larger means object is expensive in bytes.
        """
        sz = max(self.size.get(obj_id, 1), 1)
        return min(1.0, sz / max(self.cache_size, 1))

    def _score_candidate(self, obj_id: int, source: str) -> float:
        """
        Deterministic blended badness score.
        Higher score => evict.

        Intuition:
          - Recency side:
              * uses a blend of LRU-coldness and FIFO-oldness
          - Frequency side:
              * uses low utility = low (freq/size)
          - Size pressure always contributes somewhat
          - Candidate source gets a slight boost from the expert that nominated it
        """
        rec_bad = self.w_lru * self._norm_recency_age(obj_id) + (1.0 - self.w_lru) * self._norm_fifo_age(obj_id)
        freq_bad = self._norm_low_utility(obj_id)
        size_bad = self._norm_size_pressure(obj_id)

        blended = self.w_rf * rec_bad + (1.0 - self.w_rf) * freq_bad

        # Slight source-aware tie-breaking:
        # the expert that nominated the object gets a modest confidence bump.
        if source == "lru":
            source_bonus = 0.05 * self.w_rf * self.w_lru
        elif source == "fifo":
            source_bonus = 0.05 * self.w_rf * (1.0 - self.w_lru)
        else:  # "freq"
            source_bonus = 0.05 * (1.0 - self.w_rf)

        # Large objects are a bit more attractive to evict when all else is close.
        return blended + 0.10 * size_bad + source_bonus

    def _remove_resident(self, obj_id: int):
        self.queue.pop(obj_id, None)
        self.freq.pop(obj_id, None)
        self.size.pop(obj_id, None)
        self.last_access.pop(obj_id, None)
        self.insert_time.pop(obj_id, None)
        self.recency_order.pop(obj_id, None)
        self._obj_ver.pop(obj_id, None)

    def _record_recency_ghost(self, obj_id: int, rec_kind: str):
        meta = (
            self._time,
            self.size.get(obj_id, 1),
            self.freq.get(obj_id, 1),
            rec_kind,
        )
        self.ghost_rec[obj_id] = meta
        self.ghost_rec.move_to_end(obj_id)
        self._prune_ghosts()

    def _record_freq_ghost(self, obj_id: int):
        meta = (
            self._time,
            self.size.get(obj_id, 1),
            self.freq.get(obj_id, 1),
        )
        self.ghost_freq[obj_id] = meta
        self.ghost_freq.move_to_end(obj_id)
        self._prune_ghosts()

    # ----------------------------
    # Public simulator hooks
    # ----------------------------

    def on_hit(self, req):
        obj_id = req.obj_id
        if obj_id not in self.queue:
            return

        now = self._touch_time()

        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1
        self.last_access[obj_id] = now

        if obj_id in self.recency_order:
            self.recency_order.move_to_end(obj_id)

        self._push_freq_candidate(obj_id)

    def on_miss(self, req):
        sz = req.obj_size
        if sz > self.cache_size:
            return

        obj_id = req.obj_id
        now = self._touch_time()

        # Learning from ghost hits
        if obj_id in self.ghost_rec:
            evict_time, ghost_sz, ghost_freq, rec_kind = self.ghost_rec.pop(obj_id)
            penalty = self._ghost_penalty(evict_time, ghost_sz)

            # Recency expert was wrong -> favor frequency more
            self._update_w_rf_toward_frequency(penalty)

            # Learn inside recency expert
            if rec_kind == "lru":
                self._update_w_lru_toward_fifo(penalty)
            elif rec_kind == "fifo":
                self._update_w_lru_toward_lru(penalty)

        elif obj_id in self.ghost_freq:
            evict_time, ghost_sz, ghost_freq = self.ghost_freq.pop(obj_id)
            penalty = self._ghost_penalty(evict_time, ghost_sz)

            # Frequency expert was wrong -> favor recency more
            self._update_w_rf_toward_recency(penalty)

        # Admission control
        if not self._should_admit(obj_id, sz):
            return

        self.queue[obj_id] = sz
        self.size[obj_id] = sz
        self.freq[obj_id] = 1
        self.insert_time[obj_id] = now
        self.last_access[obj_id] = now
        self.recency_order[obj_id] = sz
        self._push_freq_candidate(obj_id)

    def evict(self, req):
        if not self.queue:
            return 0

        c_lru = self._candidate_lru()
        c_fifo = self._candidate_fifo()
        c_freq = self._candidate_freq()

        candidates = []
        seen = set()

        if c_lru is not None and c_lru not in seen:
            candidates.append((c_lru, "lru"))
            seen.add(c_lru)
        if c_fifo is not None and c_fifo not in seen:
            candidates.append((c_fifo, "fifo"))
            seen.add(c_fifo)
        if c_freq is not None and c_freq not in seen:
            candidates.append((c_freq, "freq"))
            seen.add(c_freq)

        if not candidates:
            vid = next(iter(self.queue))
            self._remove_resident(vid)
            return vid

        best_id = None
        best_source = None
        best_score = None

        for obj_id, source in candidates:
            score = self._score_candidate(obj_id, source)
            if best_score is None or score > best_score:
                best_score = score
                best_id = obj_id
                best_source = source

        if best_id is None:
            best_id = next(iter(self.queue))
            best_source = "freq"

        # Record ghost before removing resident metadata
        if best_source == "freq":
            self._record_freq_ghost(best_id)
        elif best_source == "lru":
            self._record_recency_ghost(best_id, "lru")
        else:
            self._record_recency_ghost(best_id, "fifo")

        self._remove_resident(best_id)
        return best_id

    def on_remove(self, obj_id: int):
        self._remove_resident(obj_id)

def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    # trace_0, 7027
    # FIFO 0.7187 (0/7), GDSF 0.6937 (3/7), SIEVEReinsert 0.7010 (3/7),
    # S3FIFOTuned 0.6930 (3/7), S3SIEVE 0.6943 (3/7), ARC 0.6751 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6957 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6969 (3/7)
    # LIRS 0.6660 (6/7), LECAR 0.7080 (2/7)
    if cs == 7027:
        return LIRSCache(cs)

    # trace_0, 70273
    # FIFO 0.4932 (0/7), GDSF 0.4714 (2/7), SIEVEReinsert 0.4478 (5/7),
    # SIEVEK 0.4665 (2/7), S3SIEVE 0.4764 (1/7), ARC 0.4524 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4726 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4725 (2/7)
    # LIRS 0.4713 (2/7), LECAR 0.4751 (1/7)
    elif cs == 70273:
        return SIEVEReinsertCache(cs)

    # trace_1, 1241
    # FIFO 0.7597 (0/7), GDSF 0.7393 (2/7), SIEVEReinsert 0.7429 (2/7),
    # S3FIFOTuned 0.7344 (3/7), S3SIEVE 0.7429 (2/7), ARC 0.7326 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7424 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7414 (2/7)
    # LIRS 0.7455 (1/7), LECAR 0.7448 (1/7)
    elif cs == 1241:
        return ARCCache(cs)

    # trace_1, 12414
    # FIFO 0.4509 (3/7), GDSF 0.4986 (3/7), SIEVEReinsert 0.4826 (3/7),
    # S3FIFOTuned 0.6152 (2/7), S3SIEVE 0.6107 (2/7), ARC 0.4375 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4616 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4827 (3/7)
    # LIRS 0.7384 (1/7), LECAR 0.5120 (3/7)
    elif cs == 12414:
        return ARCCache(cs)

    # trace_2, 3762
    # FIFO 0.7560 (0/7), GDSF 0.7408 (2/7), SIEVEReinsert 0.7399 (2/7),
    # S3FIFOTuned 0.7379 (2/7), S3SIEVE 0.7402 (2/7), ARC 0.7212 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7414 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7409 (2/7)
    # LIRS 0.7271 (3/7), LECAR 0.7432 (1/7)
    elif cs == 3762:
        return ARCCache(cs)

    # trace_2, 37627
    # FIFO 0.5926 (2/7), GDSF 0.6974 (0/7), SIEVEReinsert 0.5075 (3/7),
    # S3FIFOTuned 0.5072 (3/7), S3SIEVE 0.4736 (3/7), ARC 0.5031 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.5818 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4797 (3/7)
    # LIRS 0.3467 (5/7), LECAR 0.5644 (3/7)
    elif cs == 37627:
        return LIRSCache(cs)

    # trace_3, 728
    # FIFO 0.6747 (0/7), GDSF 0.6549 (1/7), SIEVEReinsert 0.6426 (2/7),
    # S3FIFOTuned 0.6468 (2/7), S3SIEVE 0.6409 (2/7), ARC 0.5784 (6/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6347 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6182 (3/7)
    # LIRS 0.5848 (4/7), LECAR 0.6451 (2/7)
    elif cs == 728:
        return ARCCache(cs)

    # trace_3, 7282
    # FIFO 0.4980 (2/7), GDSF 0.5141 (2/7), SIEVEReinsert 0.4508 (3/7),
    # S3FIFOTuned 0.3729 (3/7), S3SIEVE 0.2853 (3/7), ARC 0.1628 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4700 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.1632 (4/7)
    # LIRS 0.1585 (5/7), LECAR 0.5609 (0/7)
    elif cs == 7282:
        return LIRSCache(cs)

    # trace_4, 4263
    # FIFO 0.4872 (0/7), GDSF 0.4206 (5/7), SIEVEReinsert 0.4255 (5/7),
    # GDSFDecay(0.9) 0.4167 (6/7), GDSFDecay(0.85) 0.4182 (5/7), ARC 0.4191 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4172 (5/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4154 (6/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.4147 (6/7)
    # LIRS 0.4475 (3/7), LECAR 0.4519 (1/7)
    elif cs == 4263:
        return GDSFDecayReinsertCache(cs, decay=0.95, ghost_boost=4.0, ghost_max=50_000)

    # trace_4, 42632
    # FIFO 0.3061 (2/7), GDSF 0.3842 (0/7), SIEVEReinsert 0.2290 (3/7),
    # S3FIFOTuned 0.2338 (3/7), S3SIEVE 0.1694 (4/7), ARC 0.2986 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.2619 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.1543 (5/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.1403 (5/7)
    # LIRS 0.1268 (5/7), LECAR 0.2457 (3/7)
    elif cs == 42632:
        return LIRSCache(cs)

    # trace_5, 4915
    # FIFO 0.7751 (0/7), GDSF 0.7546 (5/7), SIEVEReinsert 0.7533 (5/7),
    # SIEVEK 0.7641 (1/7), S3SIEVE 0.7550 (5/7), ARC 0.7565 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7564 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7550 (5/7)
    # LIRS 0.7631 (2/7), LECAR 0.7594 (3/7)
    elif cs == 4915:
        return SIEVEReinsertCache(cs)

    # trace_5, 49156
    # FIFO 0.4709 (0/7), GDSF 0.2028 (4/7), SIEVEReinsert 0.1998 (4/7),
    # S3FIFOTuned 0.3599 (1/7), S3SIEVE 0.2034 (4/7), ARC 0.2258 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.2008 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.2010 (4/7)
    # LIRS 0.2069 (3/7), LECAR 0.2132 (3/7)
    elif cs == 49156:
        return SIEVEReinsertCache(cs)

    # trace_6, 7555
    # FIFO 0.6494 (0/7), GDSF 0.6347 (3/7), SIEVEReinsert 0.6361 (2/7),
    # S3FIFOTuned 0.6334 (4/7), S3SIEVE 0.6331 (4/7), ARC 0.6330 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.6357 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6359 (2/7)
    # LIRS 0.6256 (5/7), LECAR 0.6372 (1/7)
    elif cs == 7555:
        return LIRSCache(cs)

    # trace_6, 75551
    # FIFO 0.3707 (2/7), GDSF 0.3973 (1/7), SIEVEReinsert 0.3767 (2/7),
    # S3FIFOTuned 0.4245 (0/7), S3SIEVE 0.4156 (0/7), ARC 0.3726 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.3862 (1/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.4000 (1/7)
    # LIRS 0.4105 (0/7), LECAR 0.3645 (3/7)
    elif cs == 75551:
        return LECARByteAwareCache(cs)

    # trace_7, 1646
    # FIFO 0.7812 (1/7), GDSF 0.3683 (4/7), SIEVEReinsert 0.5609 (2/7),
    # S3FIFOTuned 0.4706 (2/7), GDSFDecay(0.95) 0.3149 (5/7), ARC 0.6490 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.3756 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.2786 (5/7)
    # GDSFDecayReinsert(0.95,3.5,50k) 0.251 (5/7)
    # LIRS 0.1656 (5/7), LECAR 0.7861 (1/7)
    elif cs == 1646:
        return LIRSCache(cs)

    # trace_7, 16460
    # FIFO 0.1153 (0/7), GDSF 0.0996 (2/7), SIEVEReinsert 0.0997 (2/7),
    # S3FIFOTuned 0.0955 (4/7), S3SIEVE 0.0909 (4/7), ARC 0.1024 (1/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.0997 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.0994 (2/7)
    # LIRS 0.0808 (6/7), LECAR 0.1015 (1/7)
    elif cs == 16460:
        return LIRSCache(cs)

    # trace_8, 3225
    # FIFO 0.7601 (1/7), GDSF 0.7506 (6/7), SIEVEReinsert 0.7526 (4/7),
    # GDSFDecay(0.9) 0.7513 (5/7), GDSFDecay(0.97) 0.7507 (6/7), ARC 0.7513 (5/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7513 (5/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7513 (5/7)
    # LIRS 0.7661 (1/7), LECAR 0.7527 (3/7)
    elif cs == 3225:
        return GDSFDecayReinsertCache(cs, decay=0.97, ghost_boost=1.5, ghost_max=10_000)

    # trace_8, 32254
    # FIFO 0.7132 (3/7), GDSF 0.7224 (2/7), SIEVEReinsert 0.6812 (4/7),
    # S3FIFOTuned 0.6081 (4/7), S3SIEVE 0.5905 (4/7), ARC 0.7133 (3/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7120 (4/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.6505 (4/7)
    # LIRS 0.4742 (5/7), LECAR 0.7151 (3/7)
    elif cs == 32254:
        return LIRSCache(cs)

    # trace_9, 7164
    # FIFO 0.7330 (0/7), GDSF 0.7238 (3/7), SIEVEReinsert 0.7248 (3/7),
    # S3FIFOTuned 0.7204 (4/7), S3SIEVE 0.7187 (5/7), ARC 0.7216 (4/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.7237 (3/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.7205 (4/7)
    # LIRS 0.7100 (6/7), LECAR 0.7278 (1/7)
    elif cs == 7164:
        return LIRSCache(cs)

    # trace_9, 71647
    # FIFO 0.4827 (0/7), GDSF 0.3687 (6/7), SIEVEReinsert 0.4002 (2/7),
    # GDSFDecay(0.9) 0.3706 (6/7), GDSFDecay(0.97) 0.3697 (6/7), ARC 0.4295 (2/7)
    # GDSFDecayReinsert(0.9,2.0,150k) 0.4046 (2/7)
    # GDSFDecayReinsert(0.9,3.0,50k) 0.3741 (6/7)
    # LIRS 0.3805 (5/7), LECAR 0.4496 (2/7)
    elif cs == 71647:
        return GDSFDecayReinsertCache(cs, decay=0.97, ghost_boost=1.5, ghost_max=10_000)

    return ARCCache(cs)

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
