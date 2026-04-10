from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request
import math


class GDSFDecayReinsertCache:
    """
    GDSF with exponential frequency decay, ghost list, ghost-hit boost.
    Score: H(x) = L + freq(x) / size(x)
    """

    def __init__(self, cache_size, decay=0.92, ghost_boost=2.5, ghost_max=100_000):
        import heapq
        self.cache_size = cache_size
        self.decay = decay
        self.ghost_boost = ghost_boost
        self.queue = {}
        self.heap = []
        self.L = 0.0
        self._ver = 0
        self._heapq = heapq
        self.G_q = deque()
        self.G_s = set()
        self.G_max = ghost_max

    def _ghost_add(self, obj_id):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        while len(self.G_s) > self.G_max and self.G_q:
            self.G_s.discard(self.G_q.popleft())

    def _push(self, obj_id, size, freq):
        self._ver += 1
        H = self.L + (freq / max(size, 1))
        self.queue[obj_id] = (size, freq, H, self._ver)
        self._heapq.heappush(self.heap, (H, self._ver, obj_id))

    def on_hit(self, req):
        rec = self.queue.get(req.obj_id)
        if rec is None:
            return
        size, freq, _, _ = rec
        size = req.obj_size or size
        freq = freq * self.decay + 1.0
        self._push(req.obj_id, size, freq)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, size = req.obj_id, req.obj_size
        if obj_id in self.G_s:
            self.G_s.discard(obj_id)
            self._push(obj_id, size, self.ghost_boost)
        else:
            self._push(obj_id, size, 1.0)

    def evict(self, req):
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
        obj_id = next(iter(self.queue))
        self.queue.pop(obj_id)
        self._ghost_add(obj_id)
        return obj_id

    def on_remove(self, obj_id):
        self.queue.pop(obj_id, None)

class SIEVEReinsertCache:
    """SIEVE with ghost-hit reinsertion protection."""

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "visited")
        def __init__(self, obj_id, visited=False):
            self.obj_id = obj_id
            self.prev = self.next = None
            self.visited = visited

    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = {}
        self._head = self._Node(-1)
        self._tail = self._Node(-2)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hand = None
        self._ghost_max = 100_000
        self._ghost_q = deque()
        self._ghost_s = set()

    def _ghost_add(self, obj_id):
        self._ghost_s.add(obj_id)
        self._ghost_q.append(obj_id)
        while len(self._ghost_s) > self._ghost_max and self._ghost_q:
            self._ghost_s.discard(self._ghost_q.popleft())

    def _insert_head(self, node):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def on_hit(self, req):
        node = self.queue.get(req.obj_id)
        if node is not None:
            node.visited = True

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        protect = req.obj_id in self._ghost_s
        if protect:
            self._ghost_s.discard(req.obj_id)
        node = self._Node(req.obj_id, visited=protect)
        self.queue[req.obj_id] = node
        self._insert_head(node)
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None

    def evict(self, req):
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

    def on_remove(self, obj_id):
        node = self.queue.pop(obj_id, None)
        if node is None:
            return
        if self._hand is node:
            self._hand = node.prev if node.prev is not self._head else self._tail.prev
        self._remove_node(node)
        if not self.queue:
            self._hand = None

class ARCCache:
    """ARC in byte capacity form."""

    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.T1 = {}
        self.T2 = {}
        self.B1_q = deque()
        self.B1_s = set()
        self.B2_q = deque()
        self.B2_s = set()
        self.p = 0
        self.queue = {}
        self.t1_bytes = 0
        self.t2_bytes = 0

    def _ghost_prune(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
            self.B2_q.popleft()

    def _ghost_add_B1(self, obj_id):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._ghost_prune()

    def _ghost_add_B2(self, obj_id):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._ghost_prune()

    def _ghost_remove(self, obj_id):
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def on_hit(self, req):
        obj_id = req.obj_id
        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        elif obj_id in self.T2:
            sz = self.T2.pop(obj_id)
            self.T2[obj_id] = sz
        else:
            return

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
        else:
            self.T1[obj_id] = sz
            self.t1_bytes += sz
        self.queue[obj_id] = sz

    def _arc_adjust_p(self, req):
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

    def evict(self, req):
        if not self.queue:
            return 0
        self._arc_adjust_p(req)
        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B1(victim_id)
            if len(self.B1_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune()
                if self.B1_q:
                    self.B1_s.discard(self.B1_q.popleft())
            return victim_id
        if self.T2:
            victim_id, victim_sz = next(iter(self.T2.items()))
            self.T2.pop(victim_id)
            self.t2_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B2(victim_id)
            if len(self.B2_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune()
                if self.B2_q:
                    self.B2_s.discard(self.B2_q.popleft())
            return victim_id
        victim_id = next(iter(self.queue))
        self.queue.pop(victim_id)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id):
        if obj_id in self.T1:
            self.t1_bytes -= self.T1.pop(obj_id)
        if obj_id in self.T2:
            self.t2_bytes -= self.T2.pop(obj_id)
        self.queue.pop(obj_id, None)

class LIRSCache:
    """Simplified LIRS with tunable LIR ratio."""

    def __init__(self, cache_size, lir_ratio=0.99):
        self.cache_size = cache_size
        self.lir_size = max(1, int(cache_size * lir_ratio))
        self.hir_size = max(1, cache_size - self.lir_size)
        self.status = {}
        self.stack = OrderedDict()
        self.hir_list = OrderedDict()
        self.lir_bytes = 0
        self.hir_bytes = 0
        self.queue = {}

    def _stack_prune(self):
        while self.stack:
            obj_id, is_lir = next(iter(self.stack.items()))
            if is_lir:
                break
            self.stack.pop(obj_id, None)

    def _demote_lir_bottom(self):
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

    def on_hit(self, req):
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
                self.stack.pop(obj_id)
                self.status[obj_id] = ("LIR", sz)
                self.hir_list.pop(obj_id, None)
                self.hir_bytes -= sz
                self.lir_bytes += sz
                self.stack[obj_id] = True
                while self.lir_bytes > self.lir_size:
                    self._demote_lir_bottom()
            else:
                self.hir_list.pop(obj_id, None)
                self.hir_list[obj_id] = sz
                self.stack[obj_id] = False

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        if obj_id in self.stack:
            self.stack.pop(obj_id)
            self.status[obj_id] = ("LIR", sz)
            self.lir_bytes += sz
            self.stack[obj_id] = True
            while self.lir_bytes > self.lir_size:
                self._demote_lir_bottom()
        else:
            self.status[obj_id] = ("HIR_RES", sz)
            self.hir_list[obj_id] = sz
            self.hir_bytes += sz
            self.stack[obj_id] = False
        self.queue[obj_id] = sz

    def evict(self, req):
        if not self.queue:
            return 0
        if self.hir_list:
            vid, vsz = self.hir_list.popitem(last=False)
            self.hir_bytes -= vsz
            if vid in self.stack:
                self.status[vid] = ("HIR_NONRES", vsz)
            else:
                self.status.pop(vid, None)
            self.queue.pop(vid, None)
            return vid
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

    def on_remove(self, obj_id):
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
        while len(self.stack) > max(len(self.queue) * 3, 10000):
            bot_id = next(iter(self.stack))
            self.stack.pop(bot_id)
            info2 = self.status.get(bot_id)
            if info2 and info2[0] == "HIR_NONRES":
                self.status.pop(bot_id, None)

class CACHEUSCache:
    """
    CACHEUS-inspired adaptive cache.

    Dynamically blends between a scan-resistant expert (SIEVE/CLOCK-like)
    and a churn-resistant expert (LFU-like) using reinforcement learning
    on ghost hits, similar to LeCaR but with experts that better match
    real workload patterns.

    Expert 1 (SR - Scan Resistant): CLOCK with visited bit.
      Good when: working set is stable, scans pollute.
    Expert 2 (CR - Churn Resistant): LFU via min-heap on freq/size.
      Good when: popularity shifts slowly, frequency is a good signal.

    Learning: ghost lists track recently evicted items per-expert.
    Ghost hits from expert X's list → decrease X's weight (it made a mistake).
    """

    class _CNode:
        __slots__ = ("obj_id", "prev", "next", "visited", "freq", "size")
        def __init__(self, obj_id, size, visited=False):
            self.obj_id = obj_id
            self.size = size
            self.prev = self.next = None
            self.visited = visited
            self.freq = 1

    def __init__(self, cache_size, learning_rate=0.45, discount=0.005):
        import heapq
        self.cache_size = cache_size
        self.lr = learning_rate
        self.discount = discount
        self._heapq = heapq

        # Shared state
        self.queue = {}        # obj_id -> _CNode
        self.freq_map = {}     # obj_id -> freq (for heap)
        self.sizes_map = {}    # obj_id -> size

        # Expert 1: CLOCK (scan-resistant)
        self._head = self._CNode(-1, 0)
        self._tail = self._CNode(-2, 0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hand = None

        # Expert 2: LFU heap
        self.heap = []
        self._ver = 0

        # Weight: w = probability of using SR expert
        self.w = 0.5
        self._credit = 0.0

        # Ghost lists
        self.ghost_sr = OrderedDict()  # obj_id -> eviction_time
        self.ghost_cr = OrderedDict()
        self.ghost_max = 100_000
        self._time = 0

    def _insert_head(self, node):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _lfu_push(self, obj_id):
        f = self.freq_map.get(obj_id, 1)
        sz = max(self.sizes_map.get(obj_id, 1), 1)
        self._ver += 1
        self._heapq.heappush(self.heap, (f / sz, self._ver, obj_id))

    def on_hit(self, req):
        obj_id = req.obj_id
        node = self.queue.get(obj_id)
        if node is None:
            return
        self._time += 1
        node.visited = True
        node.freq += 1
        self.freq_map[obj_id] = node.freq
        self._lfu_push(obj_id)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        self._time += 1

        # Learn from ghost hits
        if obj_id in self.ghost_sr:
            evict_time = self.ghost_sr.pop(obj_id)
            age = max(self._time - evict_time, 1)
            d = max(math.pow(1 - self.discount, age), 0.01)
            # SR evicted this but it came back → favor CR
            self.w = max(0.001, self.w * math.exp(-self.lr * d))
        elif obj_id in self.ghost_cr:
            evict_time = self.ghost_cr.pop(obj_id)
            age = max(self._time - evict_time, 1)
            d = max(math.pow(1 - self.discount, age), 0.01)
            # CR evicted this but it came back → favor SR
            self.w = min(0.999, 1.0 - (1.0 - self.w) * math.exp(-self.lr * d))

        node = self._CNode(obj_id, sz)
        self.queue[obj_id] = node
        self.freq_map[obj_id] = 1
        self.sizes_map[obj_id] = sz
        self._insert_head(node)
        self._lfu_push(obj_id)
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None

    def evict(self, req):
        if not self.queue:
            return 0
        self._credit += self.w
        if self._credit >= 1.0:
            self._credit -= 1.0
            vid = self._evict_sr()
            if vid is not None:
                self.ghost_sr[vid] = self._time
                if len(self.ghost_sr) > self.ghost_max:
                    self.ghost_sr.popitem(last=False)
                return vid
            vid = self._evict_cr()
            if vid is not None:
                return vid
        else:
            vid = self._evict_cr()
            if vid is not None:
                self.ghost_cr[vid] = self._time
                if len(self.ghost_cr) > self.ghost_max:
                    self.ghost_cr.popitem(last=False)
                return vid
            vid = self._evict_sr()
            if vid is not None:
                return vid
        vid = next(iter(self.queue))
        self._cleanup(vid)
        return vid

    def _evict_sr(self):
        """CLOCK-style eviction: find unvisited victim."""
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None
            if self._hand is None:
                return None
        node = self._hand
        limit = len(self.queue) * 2 + 2
        i = 0
        while i < limit:
            i += 1
            if node is self._head:
                node = self._tail.prev
                continue
            if node is self._tail:
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
            self.freq_map.pop(vid, None)
            self.sizes_map.pop(vid, None)
            if not self.queue:
                self._hand = None
            return vid
        return None

    def _evict_cr(self):
        """LFU-style eviction from heap."""
        while self.heap:
            score, ver, obj_id = self.heap[0]
            if obj_id not in self.queue:
                self._heapq.heappop(self.heap)
                continue
            cur_f = self.freq_map.get(obj_id, 0)
            cur_sz = max(self.sizes_map.get(obj_id, 1), 1)
            cur_score = cur_f / cur_sz
            if abs(cur_score - score) > 1e-9:
                self._heapq.heappop(self.heap)
                continue
            self._heapq.heappop(self.heap)
            node = self.queue.pop(obj_id, None)
            if node:
                self._remove_node(node)
                if self._hand is node:
                    self._hand = self._tail.prev if self._tail.prev is not self._head else None
            self.freq_map.pop(obj_id, None)
            self.sizes_map.pop(obj_id, None)
            if not self.queue:
                self._hand = None
            return obj_id
        return None

    def _cleanup(self, vid):
        node = self.queue.pop(vid, None)
        if node:
            self._remove_node(node)
        self.freq_map.pop(vid, None)
        self.sizes_map.pop(vid, None)
        if not self.queue:
            self._hand = None

    def on_remove(self, obj_id):
        node = self.queue.pop(obj_id, None)
        if node:
            if self._hand is node:
                self._hand = node.prev if node.prev is not self._head else self._tail.prev
            self._remove_node(node)
        self.freq_map.pop(obj_id, None)
        self.sizes_map.pop(obj_id, None)
        if not self.queue:
            self._hand = None

class ARCGhostBoostCache:
    """
    ARC variant with aggressive ghost utilization.

    Changes from standard ARC:
    - Larger ghost lists (proportional to cache size)
    - On B2 ghost hit, insert into T2 at MRU position with
      "warm" status (already proven frequent)
    - Slightly biased initial p toward recency (helps
      traces where new items are likely to be re-accessed)
    """

    def __init__(self, cache_size, initial_p_ratio=0.5):
        self.cache_size = cache_size
        self.T1 = {}
        self.T2 = {}
        self.B1_q = deque()
        self.B1_s = set()
        self.B2_q = deque()
        self.B2_s = set()
        self.p = int(cache_size * initial_p_ratio)
        self.queue = {}
        self.t1_bytes = 0
        self.t2_bytes = 0
        self._ghost_max = 200_000

    def _ghost_prune_B1(self):
        while len(self.B1_s) > self._ghost_max and self.B1_q:
            self.B1_s.discard(self.B1_q.popleft())

    def _ghost_prune_B2(self):
        while len(self.B2_s) > self._ghost_max and self.B2_q:
            self.B2_s.discard(self.B2_q.popleft())

    def on_hit(self, req):
        obj_id = req.obj_id
        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        elif obj_id in self.T2:
            sz = self.T2.pop(obj_id)
            self.T2[obj_id] = sz

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self.B1_s.discard(obj_id)
            self.B2_s.discard(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
        else:
            self.T1[obj_id] = sz
            self.t1_bytes += sz
        self.queue[obj_id] = sz

    def evict(self, req):
        if not self.queue:
            return 0
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
        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self.B1_s.add(victim_id)
            self.B1_q.append(victim_id)
            self._ghost_prune_B1()
            return victim_id
        if self.T2:
            victim_id, victim_sz = next(iter(self.T2.items()))
            self.T2.pop(victim_id)
            self.t2_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self.B2_s.add(victim_id)
            self.B2_q.append(victim_id)
            self._ghost_prune_B2()
            return victim_id
        victim_id = next(iter(self.queue))
        self.queue.pop(victim_id)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id):
        if obj_id in self.T1:
            self.t1_bytes -= self.T1.pop(obj_id)
        if obj_id in self.T2:
            self.t2_bytes -= self.T2.pop(obj_id)
        self.queue.pop(obj_id, None)

class AdaptiveWTinyLFUCache:
    """
    Adaptive W-TinyLFU / SLRU hybrid.

    Designed for traces where the current winners only beat ~3-4 baselines:
      - many one-hit or weakly-reused objects
      - some short recency bursts
      - changing phases where recency-vs-frequency shifts over time

    Structure:
      - window: small LRU segment for new arrivals / bursts
      - probation: main-cache probationary segment
      - protected: main-cache protected segment for repeated hits

    Adaptation:
      - exact bounded recent-frequency history (TinyLFU-like, but exact over a finite window)
      - ghost_window hit   => grow window  (recency matters more)
      - ghost_main hit     => shrink window (frequency/protection matters more)

    Eviction:
      - when window overflows, compare window-LRU candidate vs probation-LRU victim
        using score = est_freq / size^alpha
      - reject the weaker one
    """

    def __init__(
        self,
        cache_size: int,
        window_frac: float = 0.08,
        protected_frac: float = 0.80,
        history_max: int = 200_000,
        ghost_max: int = 100_000,
        size_exp: float = 0.50,
    ):
        self.cache_size = cache_size
        self.window_frac = window_frac
        self.protected_frac = protected_frac
        self.history_max = history_max
        self.ghost_max = ghost_max
        self.size_exp = size_exp

        self.window_target = max(1, int(cache_size * window_frac))
        self.min_window = max(1, int(cache_size * 0.02))
        self.max_window = max(self.min_window, int(cache_size * 0.50))

        # Resident segments (LRU at front, MRU at end)
        self.window = OrderedDict()      # obj_id -> size
        self.probation = OrderedDict()   # obj_id -> size
        self.protected = OrderedDict()   # obj_id -> size

        # obj_id -> "W" | "P" | "R"
        self.where = {}
        self.queue = {}  # required by plugin boilerplate

        self.window_bytes = 0
        self.probation_bytes = 0
        self.protected_bytes = 0

        # Exact bounded recent-history frequency
        self.hist_q = deque()
        self.hist_cnt = {}

        # Ghosts for adaptation
        self.ghost_window = OrderedDict()
        self.ghost_main = OrderedDict()

    def _record_access(self, obj_id: int):
        self.hist_q.append(obj_id)
        self.hist_cnt[obj_id] = self.hist_cnt.get(obj_id, 0) + 1

        while len(self.hist_q) > self.history_max:
            old = self.hist_q.popleft()
            c = self.hist_cnt.get(old, 0)
            if c <= 1:
                self.hist_cnt.pop(old, None)
            else:
                self.hist_cnt[old] = c - 1

    def _ghost_add(self, ghost: OrderedDict, obj_id: int):
        ghost.pop(obj_id, None)
        ghost[obj_id] = None
        while len(ghost) > self.ghost_max:
            ghost.popitem(last=False)

    def _adapt_window(self, obj_id: int, obj_size: int):
        step = max(obj_size, max(1, self.cache_size // 100))

        if obj_id in self.ghost_window:
            self.ghost_window.pop(obj_id, None)
            self.window_target = min(self.max_window, self.window_target + step)
        elif obj_id in self.ghost_main:
            self.ghost_main.pop(obj_id, None)
            self.window_target = max(self.min_window, self.window_target - step)

    def _score(self, obj_id: int, size: int) -> float:
        freq = self.hist_cnt.get(obj_id, 0)
        if freq <= 0:
            freq = 1
        return freq / (max(size, 1) ** self.size_exp)

    def _main_capacity(self) -> int:
        return max(1, self.cache_size - self.window_target)

    def _protected_target(self) -> int:
        return max(1, int(self._main_capacity() * self.protected_frac))

    def _rebalance_protected(self):
        target = self._protected_target()
        while self.protected_bytes > target and self.protected:
            obj_id, sz = self.protected.popitem(last=False)  # LRU protected -> probation
            self.protected_bytes -= sz
            self.probation[obj_id] = sz
            self.probation_bytes += sz
            self.where[obj_id] = "P"

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        self._record_access(obj_id)

        loc = self.where.get(obj_id)
        if loc is None:
            return

        if loc == "W":
            self.window.move_to_end(obj_id)

        elif loc == "P":
            sz = self.probation.pop(obj_id)
            self.probation_bytes -= sz
            self.protected[obj_id] = sz
            self.protected_bytes += sz
            self.where[obj_id] = "R"
            self._rebalance_protected()

        elif loc == "R":
            self.protected.move_to_end(obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        sz = req.obj_size

        self._record_access(obj_id)
        self._adapt_window(obj_id, sz)

        # Defensive: should not already exist on a miss, but keep logic safe.
        if obj_id in self.queue:
            return

        self.window[obj_id] = sz
        self.window_bytes += sz
        self.where[obj_id] = "W"
        self.queue[obj_id] = sz

    def _evict_from_window(self):
        obj_id, sz = self.window.popitem(last=False)
        self.window_bytes -= sz
        self.where.pop(obj_id, None)
        self.queue.pop(obj_id, None)
        self._ghost_add(self.ghost_window, obj_id)
        return obj_id

    def _evict_from_probation(self):
        obj_id, sz = self.probation.popitem(last=False)
        self.probation_bytes -= sz
        self.where.pop(obj_id, None)
        self.queue.pop(obj_id, None)
        self._ghost_add(self.ghost_main, obj_id)
        return obj_id

    def _evict_from_protected(self):
        obj_id, sz = self.protected.popitem(last=False)
        self.protected_bytes -= sz
        self.where.pop(obj_id, None)
        self.queue.pop(obj_id, None)
        self._ghost_add(self.ghost_main, obj_id)
        return obj_id

    def evict(self, req: Request):
        if not self.queue:
            return 0

        self._rebalance_protected()

        # Admission decision point: window overflow
        if self.window and self.window_bytes > self.window_target:
            cand_id, cand_sz = next(iter(self.window.items()))

            if self.probation:
                vict_id, vict_sz = next(iter(self.probation.items()))

                cand_score = self._score(cand_id, cand_sz)
                vict_score = self._score(vict_id, vict_sz)

                if cand_score >= vict_score:
                    # Admit candidate into main, evict probation victim
                    self.window.pop(cand_id, None)
                    self.window_bytes -= cand_sz

                    self.probation[cand_id] = cand_sz
                    self.probation_bytes += cand_sz
                    self.where[cand_id] = "P"

                    self.probation.pop(vict_id, None)
                    self.probation_bytes -= vict_sz
                    self.where.pop(vict_id, None)
                    self.queue.pop(vict_id, None)
                    self._ghost_add(self.ghost_main, vict_id)
                    return vict_id
                else:
                    # Reject candidate
                    return self._evict_from_window()

            # No probation victim to compare against yet: reject window tail
            return self._evict_from_window()

        # Standard fallback order:
        # probation first, then window, then protected
        if self.probation:
            return self._evict_from_probation()

        if self.window:
            return self._evict_from_window()

        if self.protected:
            return self._evict_from_protected()

        # Rare fallback
        obj_id = next(iter(self.queue.keys()))
        self.queue.pop(obj_id, None)
        self.where.pop(obj_id, None)
        self.window.pop(obj_id, None)
        self.probation.pop(obj_id, None)
        self.protected.pop(obj_id, None)
        return obj_id

    def on_remove(self, obj_id: int):
        loc = self.where.pop(obj_id, None)
        if loc == "W":
            sz = self.window.pop(obj_id, None)
            if sz is not None:
                self.window_bytes -= sz
        elif loc == "P":
            sz = self.probation.pop(obj_id, None)
            if sz is not None:
                self.probation_bytes -= sz
        elif loc == "R":
            sz = self.protected.pop(obj_id, None)
            if sz is not None:
                self.protected_bytes -= sz

        self.queue.pop(obj_id, None)

class ARCRestrictedCache:
    """
    ARC variant for very small, high-pressure caches.

    Changes vs ARC:
      1. Delayed promotion:
         - T1 hit does not immediately promote to T2.
         - Require two resident hits in T1 before promotion.
      2. Lightweight admission filter:
         - Maintain bounded recent request counts.
         - Cold first-touch objects enter T1 as weak probation;
           only ghost hits or repeated touches get stronger treatment.
      3. Smoother p adaptation:
         - adjust p with smaller byte steps to avoid instability in tiny caches.

    Intended use:
      traces where ARC is the right family, but small-cache ARC over-admits
      and over-promotes.
    """

    def __init__(self, cache_size: int, history_max: int = 50_000):
        from collections import deque

        self.cache_size = cache_size

        # Resident lists: LRU at front, MRU at end
        self.T1: OrderedDict[int, int] = OrderedDict()
        self.T2: OrderedDict[int, int] = OrderedDict()

        # Ghosts
        self.B1_q = deque()
        self.B1_s: set[int] = set()
        self.B2_q = deque()
        self.B2_s: set[int] = set()

        # Per-object resident hit count while in T1
        self.t1_hits: dict[int, int] = {}

        # Recent request history for admission filtering
        self.hist_q = deque()
        self.hist_cnt: dict[int, int] = {}
        self.history_max = history_max

        # Target size for T1 in bytes
        self.p = max(0, cache_size // 8)

        self.queue: dict[int, int] = {}
        self.t1_bytes = 0
        self.t2_bytes = 0

    def _record_access(self, obj_id: int):
        self.hist_q.append(obj_id)
        self.hist_cnt[obj_id] = self.hist_cnt.get(obj_id, 0) + 1
        while len(self.hist_q) > self.history_max:
            old = self.hist_q.popleft()
            c = self.hist_cnt.get(old, 0)
            if c <= 1:
                self.hist_cnt.pop(old, None)
            else:
                self.hist_cnt[old] = c - 1

    def _ghost_prune(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
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
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def _move_to_mru(self, d: OrderedDict, obj_id: int):
        d.move_to_end(obj_id)

    def _adapt_p(self, req: Request):
        obj_id = req.obj_id
        step = max(1, min(req.obj_size, max(1, self.cache_size // 32)))

        if obj_id in self.B1_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b2 // b1, 1)
            self.p = min(self.cache_size, self.p + delta * step)

        elif obj_id in self.B2_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b1 // b2, 1)
            self.p = max(0, self.p - delta * step)

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        self._record_access(obj_id)

        if obj_id in self.T1:
            self.t1_hits[obj_id] = self.t1_hits.get(obj_id, 0) + 1

            # Delayed promotion: require 2 hits while resident in T1
            if self.t1_hits[obj_id] >= 2:
                sz = self.T1.pop(obj_id)
                self.t1_bytes -= sz
                self.T2[obj_id] = sz
                self.t2_bytes += sz
                self.t1_hits.pop(obj_id, None)
            else:
                self._move_to_mru(self.T1, obj_id)

        elif obj_id in self.T2:
            self._move_to_mru(self.T2, obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        sz = req.obj_size
        self._record_access(obj_id)

        # Ghost hit => adapt and admit strongly to T2
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
            self.t1_hits.pop(obj_id, None)
            return

        # Cold miss admission filter:
        # still admit, but weakly into T1 with zero hit credit
        # This preserves ARC structure while making promotion harder.
        self.T1[obj_id] = sz
        self.t1_bytes += sz
        self.queue[obj_id] = sz
        self.t1_hits[obj_id] = 0

    def evict(self, req: Request):
        if not self.queue:
            return 0

        self._adapt_p(req)

        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id, None)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self.t1_hits.pop(victim_id, None)
            self._ghost_add_B1(victim_id)
            if len(self.B1_s) > 2 * (len(self.queue) + 1):
                self._ghost_prune()
                if self.B1_q:
                    old = self.B1_q.popleft()
                    self.B1_s.discard(old)
            return victim_id

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

        victim_id = next(iter(self.queue.keys()))
        self.queue.pop(victim_id, None)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        self.t1_hits.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id: int):
        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
        if obj_id in self.T2:
            sz = self.T2.pop(obj_id)
            self.t2_bytes -= sz
        self.t1_hits.pop(obj_id, None)
        self.queue.pop(obj_id, None)

class ARCSmoothedCache:
    """
    ARC with smoother byte-based adaptation.

    Intended for traces where standard ARC is the right family,
    but small byte-capacity caches make p adaptation too jumpy.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        self.T1 = OrderedDict()
        self.T2 = OrderedDict()

        self.B1_q = deque()
        self.B1_s = set()
        self.B2_q = deque()
        self.B2_s = set()

        self.p = max(0, cache_size // 4)

        self.queue = {}
        self.t1_bytes = 0
        self.t2_bytes = 0

        self._ghost_max = 200_000

    def _ghost_prune(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
            self.B2_q.popleft()

        while len(self.B1_s) > self._ghost_max and self.B1_q:
            self.B1_s.discard(self.B1_q.popleft())
        while len(self.B2_s) > self._ghost_max and self.B2_q:
            self.B2_s.discard(self.B2_q.popleft())

    def _ghost_add_B1(self, obj_id: int):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._ghost_prune()

    def _ghost_add_B2(self, obj_id: int):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._ghost_prune()

    def _ghost_remove(self, obj_id: int):
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def _adjust_p(self, req: Request):
        obj_id = req.obj_id

        # Fixed/smoothed step, not proportional to full object size
        step = max(1, self.cache_size // 32)

        if obj_id in self.B1_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b2 // b1, 1)
            self.p = min(self.cache_size, self.p + delta * step)

        elif obj_id in self.B2_s:
            b1 = max(len(self.B1_s), 1)
            b2 = max(len(self.B2_s), 1)
            delta = max(b1 // b2, 1)
            self.p = max(0, self.p - delta * step)

    def on_hit(self, req: Request):
        obj_id = req.obj_id

        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz

        elif obj_id in self.T2:
            self.T2.move_to_end(obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return

        obj_id = req.obj_id
        sz = req.obj_size

        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
        else:
            self.T1[obj_id] = sz
            self.t1_bytes += sz

        self.queue[obj_id] = sz

    def evict(self, req: Request):
        if not self.queue:
            return 0

        self._adjust_p(req)

        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            victim_id, victim_sz = next(iter(self.T1.items()))
            self.T1.pop(victim_id)
            self.t1_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B1(victim_id)
            return victim_id

        if self.T2:
            victim_id, victim_sz = next(iter(self.T2.items()))
            self.T2.pop(victim_id)
            self.t2_bytes -= victim_sz
            self.queue.pop(victim_id, None)
            self._ghost_add_B2(victim_id)
            return victim_id

        victim_id = next(iter(self.queue))
        self.queue.pop(victim_id, None)
        self.T1.pop(victim_id, None)
        self.T2.pop(victim_id, None)
        return victim_id

    def on_remove(self, obj_id: int):
        if obj_id in self.T1:
            self.t1_bytes -= self.T1.pop(obj_id)
        if obj_id in self.T2:
            self.t2_bytes -= self.T2.pop(obj_id)
        self.queue.pop(obj_id, None)

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

class S3FifoTunedCache:
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
            self.head = S3FifoTunedCache._Node(-1, 0)
            self.tail = S3FifoTunedCache._Node(-2, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.bytes = 0
            self.nodes: dict[int, S3FifoTunedCache._Node] = {}

        def empty(self) -> bool:
            return self.head.next is self.tail

        def push_head(self, node: "S3FifoTunedCache._Node"):
            node.next = self.head.next
            node.prev = self.head
            self.head.next.prev = node
            self.head.next = node
            self.nodes[node.obj_id] = node
            self.bytes += node.size

        def pop_tail(self) -> "S3FifoTunedCache._Node | None":
            if self.empty():
                return None
            node = self.tail.prev
            self.remove(node.obj_id)
            return node

        def remove(self, obj_id: int) -> "S3FifoTunedCache._Node | None":
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

class ARCFrequencyCache:
    """
    ARC variant that uses per-item frequency counts to pick eviction
    victims within T1/T2, rather than pure FIFO/LRU order.

    Targeted at small caches (trace_1, 1241) where ARC's structure
    is right but evicting the wrong item from T1 costs dearly.

    Changes vs standard ARC:
      - T1 and T2 are dicts (not ordered); eviction scans for
        the item with lowest freq/size score.
      - Bounded scan window (check up to 16 candidates) to limit overhead.
      - Ghost hits still drive p adaptation normally.
    """

    def __init__(self, cache_size: int, scan_window: int = 16):
        self.cache_size = cache_size
        self.scan_window = scan_window

        self.T1 = OrderedDict()   # obj_id -> size
        self.T2 = OrderedDict()   # obj_id -> size
        self.freq = {}            # obj_id -> access count

        self.B1_q = deque()
        self.B1_s = set()
        self.B2_q = deque()
        self.B2_s = set()

        self.p = 0
        self.queue = {}
        self.t1_bytes = 0
        self.t2_bytes = 0

    def _ghost_prune(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
            self.B2_q.popleft()

    def _ghost_add_B1(self, obj_id):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._ghost_prune()

    def _ghost_add_B2(self, obj_id):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._ghost_prune()

    def _ghost_remove(self, obj_id):
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def _pick_victim(self, segment, max_scan):
        """Pick item with lowest freq/size from first max_scan items."""
        best_id = None
        best_score = float('inf')
        best_sz = 0
        count = 0
        for oid, sz in segment.items():
            f = self.freq.get(oid, 0)
            score = f / max(sz, 1)
            if score < best_score or (score == best_score and count == 0):
                best_score = score
                best_id = oid
                best_sz = sz
            count += 1
            if count >= max_scan:
                break
        return best_id, best_sz

    def on_hit(self, req):
        obj_id = req.obj_id
        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1

        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        elif obj_id in self.T2:
            sz = self.T2.pop(obj_id)
            self.T2[obj_id] = sz  # move to end

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1

        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._ghost_remove(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
        else:
            self.T1[obj_id] = sz
            self.t1_bytes += sz
        self.queue[obj_id] = sz

    def evict(self, req):
        if not self.queue:
            return 0
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

        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            vid, vsz = self._pick_victim(self.T1, self.scan_window)
            if vid is not None:
                self.T1.pop(vid)
                self.t1_bytes -= vsz
                self.queue.pop(vid, None)
                self.freq.pop(vid, None)
                self._ghost_add_B1(vid)
                if len(self.B1_s) > 2 * (len(self.queue) + 1):
                    if self.B1_q:
                        self.B1_s.discard(self.B1_q.popleft())
                return vid

        if self.T2:
            vid, vsz = self._pick_victim(self.T2, self.scan_window)
            if vid is not None:
                self.T2.pop(vid)
                self.t2_bytes -= vsz
                self.queue.pop(vid, None)
                self.freq.pop(vid, None)
                self._ghost_add_B2(vid)
                if len(self.B2_s) > 2 * (len(self.queue) + 1):
                    if self.B2_q:
                        self.B2_s.discard(self.B2_q.popleft())
                return vid

        vid = next(iter(self.queue))
        self.queue.pop(vid)
        self.T1.pop(vid, None)
        self.T2.pop(vid, None)
        self.freq.pop(vid, None)
        return vid

    def on_remove(self, obj_id):
        if obj_id in self.T1:
            self.t1_bytes -= self.T1.pop(obj_id)
        if obj_id in self.T2:
            self.t2_bytes -= self.T2.pop(obj_id)
        self.queue.pop(obj_id, None)
        self.freq.pop(obj_id, None)

class LFUAgingCache:
    """
    Pure LFU with deterministic periodic halving and size-awareness.

    Targeted at trace_5, 49156 where frequency-based methods dominate
    but existing ones (GDSF, SIEVE) still only hit 4/7.

    Score = freq / size.  Every `halve_interval` misses, all freqs
    are integer-halved (freq >>= 1), preventing stale high-freq
    items from squatting.  Ghost list boosts readmitted items.

    Eviction: scan bottom of heap for lowest-score item.
    """

    def __init__(self, cache_size: int, halve_interval: int = 0, ghost_max: int = 100_000):
        import heapq
        self.cache_size = cache_size
        # Auto-tune halve interval to ~2x cache fills if not specified
        self.halve_interval = halve_interval if halve_interval > 0 else max(10000, cache_size * 2)
        self.ghost_max = ghost_max
        self._heapq = heapq

        self.queue = {}       # obj_id -> size
        self.freq = {}        # obj_id -> freq count
        self.heap = []        # (score, ver, obj_id)
        self._ver = 0
        self._miss_count = 0

        self.G_q = deque()
        self.G_s = set()
        self.G_freq = {}      # remembered freq for ghost items

    def _ghost_add(self, obj_id, freq):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        self.G_freq[obj_id] = freq
        while len(self.G_s) > self.ghost_max and self.G_q:
            old = self.G_q.popleft()
            self.G_s.discard(old)
            self.G_freq.pop(old, None)

    def _push(self, obj_id):
        self._ver += 1
        f = self.freq.get(obj_id, 1)
        sz = max(self.queue.get(obj_id, 1), 1)
        score = f / sz
        self._heapq.heappush(self.heap, (score, self._ver, obj_id))

    def _halve_all(self):
        for oid in self.freq:
            self.freq[oid] = max(1, self.freq[oid] >> 1)
        # Rebuild heap
        self.heap = []
        self._ver += 1
        for oid, sz in self.queue.items():
            f = self.freq.get(oid, 1)
            self._ver += 1
            self._heapq.heappush(self.heap, (f / max(sz, 1), self._ver, oid))

    def on_hit(self, req):
        if req.obj_id not in self.queue:
            return
        self.freq[req.obj_id] = self.freq.get(req.obj_id, 1) + 1
        self._push(req.obj_id)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size

        self._miss_count += 1
        if self._miss_count >= self.halve_interval:
            self._miss_count = 0
            self._halve_all()

        if obj_id in self.G_s:
            old_f = self.G_freq.pop(obj_id, 0)
            self.G_s.discard(obj_id)
            self.freq[obj_id] = max(old_f, 1) + 1
        else:
            self.freq[obj_id] = 1

        self.queue[obj_id] = sz
        self._push(obj_id)

    def evict(self, req):
        if not self.queue:
            return 0
        while self.heap:
            score, ver, obj_id = self.heap[0]
            if obj_id not in self.queue:
                self._heapq.heappop(self.heap)
                continue
            cur_f = self.freq.get(obj_id, 0)
            cur_sz = max(self.queue.get(obj_id, 1), 1)
            cur_score = cur_f / cur_sz
            if abs(cur_score - score) > 1e-9:
                self._heapq.heappop(self.heap)
                self._push(obj_id)
                continue
            self._heapq.heappop(self.heap)
            f = self.freq.pop(obj_id, 0)
            sz = self.queue.pop(obj_id)
            self._ghost_add(obj_id, f)
            return obj_id
        obj_id = next(iter(self.queue))
        f = self.freq.pop(obj_id, 0)
        self.queue.pop(obj_id)
        self._ghost_add(obj_id, f)
        return obj_id

    def on_remove(self, obj_id):
        self.queue.pop(obj_id, None)
        self.freq.pop(obj_id, None)

class SIEVEKReinsertCache:
    """
    SIEVE with graduated frequency counter instead of binary visited bit.

    On hit: counter = min(counter + 1, K)
    On hand pass: counter -= 1; if counter < 0, evict
    Ghost hit: reinsertion with counter = 2 (pre-warmed)

    This gives proportional protection to frequently-accessed items,
    which binary SIEVE cannot distinguish. Targeted at trace_5, 49156
    where frequency is the dominant signal but SIEVE's binary bit
    caps out at one level of protection.
    """

    class _Node:
        __slots__ = ("obj_id", "prev", "next", "counter")
        def __init__(self, obj_id, counter=0):
            self.obj_id = obj_id
            self.prev = self.next = None
            self.counter = counter

    def __init__(self, cache_size, max_counter=3, ghost_warmup=2):
        self.cache_size = cache_size
        self.K = max_counter
        self.ghost_warmup = ghost_warmup
        self.queue = {}
        self._head = self._Node(-1)
        self._tail = self._Node(-2)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hand = None
        self._ghost_max = 100_000
        self._ghost_q = deque()
        self._ghost_s = set()

    def _ghost_add(self, obj_id):
        self._ghost_s.add(obj_id)
        self._ghost_q.append(obj_id)
        while len(self._ghost_s) > self._ghost_max and self._ghost_q:
            self._ghost_s.discard(self._ghost_q.popleft())

    def _insert_head(self, node):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def on_hit(self, req):
        node = self.queue.get(req.obj_id)
        if node is not None:
            node.counter = min(node.counter + 1, self.K)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        ghost_hit = req.obj_id in self._ghost_s
        if ghost_hit:
            self._ghost_s.discard(req.obj_id)
        init_counter = self.ghost_warmup if ghost_hit else 0
        node = self._Node(req.obj_id, counter=init_counter)
        self.queue[req.obj_id] = node
        self._insert_head(node)
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None

    def evict(self, req):
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
            if node.counter > 0:
                node.counter -= 1
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

    def on_remove(self, obj_id):
        node = self.queue.pop(obj_id, None)
        if node is None:
            return
        if self._hand is node:
            self._hand = node.prev if node.prev is not self._head else self._tail.prev
        self._remove_node(node)
        if not self.queue:
            self._hand = None

class GDSFFrequencyCache:
    """
    GDSF variant with score = L + freq (no size divisor) + ghost boost.

    Hypothesis: in trace_5, 49156 size variation may be low or
    uncorrelated with reuse, so dividing by size adds noise.
    Removing size from the score focuses purely on frequency signal.
    Ghost list remembers evicted items and boosts them on readmission.
    """

    def __init__(self, cache_size, ghost_boost=3.0, ghost_max=100_000):
        import heapq
        self.cache_size = cache_size
        self.ghost_boost = ghost_boost
        self.queue = {}
        self.heap = []
        self.L = 0.0
        self._ver = 0
        self._heapq = heapq
        self.G_q = deque()
        self.G_s = set()
        self.G_max = ghost_max

    def _ghost_add(self, obj_id):
        self.G_s.add(obj_id)
        self.G_q.append(obj_id)
        while len(self.G_s) > self.G_max and self.G_q:
            self.G_s.discard(self.G_q.popleft())

    def _push(self, obj_id, size, freq):
        self._ver += 1
        H = self.L + freq
        self.queue[obj_id] = (size, freq, H, self._ver)
        self._heapq.heappush(self.heap, (H, self._ver, obj_id))

    def on_hit(self, req):
        rec = self.queue.get(req.obj_id)
        if rec is None:
            return
        size, freq, _, _ = rec
        size = req.obj_size or size
        freq = freq + 1.0
        self._push(req.obj_id, size, freq)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, size = req.obj_id, req.obj_size
        if obj_id in self.G_s:
            self.G_s.discard(obj_id)
            self._push(obj_id, size, self.ghost_boost)
        else:
            self._push(obj_id, size, 1.0)

    def evict(self, req):
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
        obj_id = next(iter(self.queue))
        self.queue.pop(obj_id)
        self._ghost_add(obj_id)
        return obj_id

    def on_remove(self, obj_id):
        self.queue.pop(obj_id, None)

def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    return GDSFFrequencyCache(cs)


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
