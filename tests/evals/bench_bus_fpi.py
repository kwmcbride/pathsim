########################################################################################
##
##                             BUS FLAT-ARRAY FPI BENCHMARK
##                               (tests/evals/bench_bus_fpi.py)
##
##                                   Kevin McBride 2026
##
########################################################################################
"""
Measures per-FPI-iteration overhead of bus block update() calls before/after the
flat-array refactor.

"Before" (dict path)  — BusSelector with _flat_indices=None uses the Python-dict
                         traversal fallback (still present in the code).
"After"  (fast path)  — BusSelector with _flat_indices injected does a single
                         numpy fancy-index op.

Timings are reported in nanoseconds per iteration.
Run with:
    uv run python tests/evals/bench_bus_fpi.py
"""

import timeit
import numpy as np
from pathsim.blocks.buses import BusCreator, BusSelector, BusMerge, BusFunction
from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Adder, Constant
from pathsim.blocks.scope import Scope


# ── helpers ─────────────────────────────────────────────────────────────────────────

def _ns(seconds, n):
    return seconds / n * 1e9


def _bar(label, ns, width=40):
    ref_ns = _bar._ref or ns
    filled = int(round(width * ns / max(ref_ns, ns)))
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label:<36s} {bar}  {ns:>8.1f} ns/iter")
_bar._ref = None


def _section(title):
    print()
    print(f"{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── 1. BusCreator.update() ────────────────────────────────────────────────────────

def bench_bus_creator(n_keys=8, n_iter=200_000):
    _section(f"BusCreator.update()  [{n_keys} scalar inputs → float64 buffer]")

    creator = BusCreator([f's{i}' for i in range(n_keys)])
    for i, k in enumerate(creator.keys):
        creator.inputs[k] = float(i)

    t = timeit.timeit(lambda: creator.update(), number=n_iter)
    ns = _ns(t, n_iter)
    _bar._ref = ns
    _bar("BusCreator (flat buffer, always)", ns)
    return ns


# ── 2. BusSelector.update() — fast vs dict ───────────────────────────────────────

def bench_bus_selector(n_keys=8, n_select=4, n_iter=200_000):
    _section(
        f"BusSelector.update()  "
        f"[buf[{n_keys}] → select {n_select} keys]"
    )

    creator = BusCreator([f's{i}' for i in range(n_keys)])
    for i, k in enumerate(creator.keys):
        creator.inputs[k] = float(i)
    creator.update()
    buf = creator.outputs['bus']

    sel_keys = [f's{i}' for i in range(n_select)]

    # ── fast path: _flat_indices injected ────────────────────────────────────────
    fast_sel = BusSelector(sel_keys)
    fast_sel._flat_indices = np.arange(n_select, dtype=np.intp)
    fast_sel.inputs['bus'] = buf

    t_fast = timeit.timeit(lambda: fast_sel.update(), number=n_iter)
    ns_fast = _ns(t_fast, n_iter)

    # ── dict fallback path (pre-refactor) ────────────────────────────────────────
    bus_dict = {f's{i}': float(i) for i in range(n_keys)}

    dict_sel = BusSelector(sel_keys)
    # _flat_indices left as None → hits dict fallback branch
    dict_sel.inputs['bus'] = bus_dict

    t_dict = timeit.timeit(lambda: dict_sel.update(), number=n_iter)
    ns_dict = _ns(t_dict, n_iter)

    ref = max(ns_fast, ns_dict)
    _bar._ref = ref
    _bar("fast path  (numpy fancy-index)", ns_fast)
    _bar("dict path  (pre-refactor)", ns_dict)
    print(f"\n  Speedup:  {ns_dict / ns_fast:.1f}×  ({ns_dict:.1f} ns → {ns_fast:.1f} ns)")
    return ns_fast, ns_dict


# ── 3. BusFunction.update() — fast vs dict ───────────────────────────────────────

def bench_bus_function(n_keys=8, n_iter=200_000):
    _section(f"BusFunction.update()  [buf[{n_keys}] → apply func → 1 output]")

    creator = BusCreator([f's{i}' for i in range(n_keys)])
    for i, k in enumerate(creator.keys):
        creator.inputs[k] = float(i)
    creator.update()
    buf = creator.outputs['bus']

    # ── fast path ──────────────────────────────────────────────────────────────
    bf_fast = BusFunction(lambda a, b: a + b, ['s0', 's1'], ['sum'])
    bf_fast._in_flat_indices = np.array([0, 1], dtype=np.intp)
    bf_fast._out_buf = np.zeros(1)
    bf_fast.inputs['bus'] = buf

    t_fast = timeit.timeit(lambda: bf_fast.update(), number=n_iter)
    ns_fast = _ns(t_fast, n_iter)

    # ── dict fallback ──────────────────────────────────────────────────────────
    bus_dict = {f's{i}': float(i) for i in range(n_keys)}
    bf_dict = BusFunction(lambda a, b: a + b, ['s0', 's1'], ['sum'])
    # Leave _in_flat_indices = None → dict fallback
    bf_dict.inputs['bus'] = bus_dict

    t_dict = timeit.timeit(lambda: bf_dict.update(), number=n_iter)
    ns_dict = _ns(t_dict, n_iter)

    ref = max(ns_fast, ns_dict)
    _bar._ref = ref
    _bar("fast path  (numpy fancy-index)", ns_fast)
    _bar("dict path  (pre-refactor)", ns_dict)
    print(f"\n  Speedup:  {ns_dict / ns_fast:.1f}×  ({ns_dict:.1f} ns → {ns_fast:.1f} ns)")
    return ns_fast, ns_dict


# ── 4. Full simulation: bus algebraic loop ───────────────────────────────────────

def bench_full_sim(n_keys=8, duration=5.0):
    """
    Algebraic loop:  Constant → BusCreator → BusSelector → Amplifier → back to BusCreator
    n_keys signals, but only 1 selected for the loop.
    """
    _section(
        f"Full sim: algebraic loop with {n_keys}-key bus  "
        f"[duration={duration}s, dt=0.01]"
    )

    def _make_sim():
        c     = Constant(1.0)
        amps  = [Amplifier(0.0) for _ in range(n_keys - 1)]   # gains=0 → convergent
        creator  = BusCreator([f's{i}' for i in range(n_keys)])
        selector = BusSelector([f's{i}' for i in range(n_keys)])
        scope = Scope(labels=[f's{i}' for i in range(n_keys)])

        conns = [Connection(c[0], creator['s0'])]
        for i, amp in enumerate(amps, start=1):
            conns.append(Connection(c[0], amp[0]))
            conns.append(Connection(amp[0], creator[f's{i}']))
        conns.append(Connection(creator[0], selector['bus']))
        for i in range(n_keys):
            conns.append(Connection(selector[f's{i}'], scope[i]))

        return Simulation([c, creator, selector, scope] + amps, conns, dt=0.01, log=False)

    # Warm-up
    sim = _make_sim()
    sim.run(duration=0.5)

    import time
    sim2 = _make_sim()
    t0 = time.perf_counter()
    sim2.run(duration=duration)
    elapsed = time.perf_counter() - t0

    n_steps = int(duration / 0.01)
    ns_per_step = elapsed / n_steps * 1e9
    print(f"  {n_keys}-key bus, {n_steps} timesteps: "
          f"{elapsed*1000:.1f} ms total, "
          f"{ns_per_step:.0f} ns/step")
    return elapsed


# ── 5. Scaling: how does cost grow with bus width? ─────────────────────────────

def bench_scaling(n_iter=100_000):
    _section("BusSelector fast-path scaling with bus width")
    print(f"  {'keys':>6}  {'ns/iter':>10}  {'ns/key':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    _bar._ref = None
    for n_keys in [2, 4, 8, 16, 32, 64]:
        creator = BusCreator([f's{i}' for i in range(n_keys)])
        for i, k in enumerate(creator.keys):
            creator.inputs[k] = float(i)
        creator.update()
        buf = creator.outputs['bus']

        sel = BusSelector([f's{i}' for i in range(n_keys)])
        sel._flat_indices = np.arange(n_keys, dtype=np.intp)
        sel.inputs['bus'] = buf

        t = timeit.timeit(lambda: sel.update(), number=n_iter)
        ns = _ns(t, n_iter)
        print(f"  {n_keys:>6}  {ns:>10.1f}  {ns/n_keys:>10.2f}")


# ── main ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import logging
    logging.getLogger('pathsim').setLevel(logging.WARNING)

    print("PathSim Bus FPI Benchmark")
    print(f"numpy {np.__version__}")
    print()

    bench_bus_creator(n_keys=8)
    bench_bus_selector(n_keys=8, n_select=4)
    bench_bus_function(n_keys=8)
    bench_scaling()
    bench_full_sim(n_keys=8, duration=3.0)

    print()
    print("Done.")
