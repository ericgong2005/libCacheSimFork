from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request

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
    return LIRSCache(common_cache_params.cache_size)


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
