from collections import deque
from libcachesim import CommonCacheParams, Request

class FifoCache:
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
