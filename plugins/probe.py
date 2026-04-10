"""
Check if next_access_vtime contains real future timestamps.
INT64_MAX (9223372036854775807) = "never accessed again" sentinel.
-1 = also commonly "no future access".
We need values that are POSITIVE and LESS than INT64_MAX.

TEST=1: Find first request with real oracle vtime (0 < val < INT64_MAX)
TEST=2: Collect 50 values and dump, distinguishing real vs sentinel
TEST=3: Dump first 20 values raw (including sentinels) to see pattern
TEST=4: Dump all field values for first request (clock_time, op, ttl, hv too)
TEST=5: Check if eviction_hook's req has different next_access_vtime than miss_hook
"""

import os
from collections import deque
from libcachesim import CommonCacheParams, Request

TEST = 1
INT64_MAX = 9223372036854775807


class ProbeCache:
    def __init__(self, cache_size: int):
        self.queue = deque()
        self.cache_size = cache_size
        self._checked = False
        self._count = 0
        self._collected = []
        self._evict_collected = []

    def _probe(self, req: Request, label: str):
        self._count += 1

        if TEST == 1:
            val = req.next_access_vtime
            if 0 < val < INT64_MAX:
                assert False, (
                    f"REAL ORACLE at req #{self._count} [{label}]: "
                    f"next_access_vtime={val}, obj_id={req.obj_id}, obj_size={req.obj_size}"
                )
            if self._count >= 2000 and not self._checked:
                self._checked = True
                assert False, f"No real oracle in first 2000 requests. All were -1, 0, or INT64_MAX."

        elif TEST == 2:
            val = req.next_access_vtime
            if label == "MISS":
                tag = "REAL" if (0 < val < INT64_MAX) else ("NEVER" if val == INT64_MAX else f"OTHER({val})")
                self._collected.append((val, tag, req.obj_id))
                if len(self._collected) >= 50 and not self._checked:
                    self._checked = True
                    real = [(v, t, oid) for v, t, oid in self._collected if t == "REAL"]
                    never = [(v, t, oid) for v, t, oid in self._collected if t == "NEVER"]
                    other = [(v, t, oid) for v, t, oid in self._collected if t not in ("REAL", "NEVER")]
                    assert False, (
                        f"50 MISS next_access_vtime: "
                        f"real={len(real)}, never={len(never)}, other={len(other)}. "
                        f"Real samples: {[(v,oid) for v,_,oid in real[:10]]}. "
                        f"Other samples: {[(v,t,oid) for v,t,oid in other[:10]]}"
                    )

        elif TEST == 3:
            val = req.next_access_vtime
            self._collected.append((label, val, req.obj_id, req.obj_size))
            if len(self._collected) >= 30 and not self._checked:
                self._checked = True
                assert False, f"First 30 values: {self._collected}"

        elif TEST == 4 and not self._checked:
            self._checked = True
            assert False, (
                f"[{label}] obj_id={req.obj_id}, obj_size={req.obj_size}, "
                f"next_access_vtime={req.next_access_vtime}, "
                f"clock_time={req.clock_time}, op={req.op}, "
                f"ttl={req.ttl}, hv={req.hv}, valid={req.valid}"
            )

        elif TEST == 5 and label == "EVICT":
            val = req.next_access_vtime
            self._evict_collected.append((val, req.obj_id, req.obj_size))
            if len(self._evict_collected) >= 20 and not self._checked:
                self._checked = True
                assert False, (
                    f"EVICT hook req values (these are the INCOMING miss request): "
                    f"{self._evict_collected}. "
                    f"Note: evict gets the INCOMING req, not the victim."
                )

    def on_hit(self, req: Request):
        self._probe(req, "HIT")

    def on_miss(self, req: Request):
        self._probe(req, "MISS")
        if req.obj_size <= self.cache_size:
            self.queue.append(req.obj_id)

    def evict(self, req: Request):
        self._probe(req, "EVICT")
        if not self.queue:
            return 0
        return self.queue.popleft()

    def on_remove(self, obj_id: int):
        try:
            self.queue.remove(obj_id)
        except ValueError:
            pass

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
    if cs == 20000000:
        return FifoCache(common_cache_params.cache_size)

    return ProbeCache(common_cache_params.cache_size)


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

    plugin_cache = PluginCache(
        cache_size=1024,  # tiny cache to force evictions fast
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
        cache_name="probe",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_cache.process_trace(reader)
