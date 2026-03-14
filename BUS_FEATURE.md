# Bus Signal Architecture — `kevin/dev/bus`

> Overview of the structured bus signal addition for `pathsim`, prepared for review prior to merge into `master`.

---

## What Was Added

Four new blocks in `src/pathsim/blocks/buses.py`, a `Bus`/`BusElement` schema type in `src/pathsim/bus.py`, and additions to `simulation.py` and `optim/booster.py`.

| Block | Role |
|-------|------|
| `BusCreator(bus)` | Packs N named scalar inputs into a pre-allocated flat `float64` ndarray |
| `BusSelector(keys)` | Extracts named keys from a bus ndarray via a single numpy fancy-index op |
| `BusMerge(n)` | Merges N bus ndarrays into one contiguous buffer; configurable conflict policy |
| `BusFunction(func, in_keys, out_keys)` | Applies a callable to selected bus signals, produces a new bus ndarray |

Bus signals are **pre-allocated flat `float64` ndarrays** at runtime — no Python dicts, no per-iteration allocation. The `Bus`/`BusElement` classes are metadata only (schema validation, docstrings, units) and are never used in the hot path.

---

## Performance Design

The primary constraint: bus signals flow through every FPI iteration of every algebraic loop they participate in. The design eliminates all per-iteration Python object overhead.

### 1. Pre-allocated flat float64 buffers

`BusCreator.__init__` pre-allocates its output buffer once:

```python
self._n = _bus_leaf_count(self.bus)           # total scalar leaves
self._buf = np.zeros(self._n)                 # float64, allocated once
self._key_layout = [(key, start, size), ...]  # pre-computed pack plan
```

`update()` writes scalars directly into the buffer by pre-computed offset — no dict creation, no key lookup:

```python
def update(self, t=None):
    for key, start, size in self._key_layout:
        val = self.inputs[key]
        if size == 1:
            try: self._buf[start] = float(val)
            except (TypeError, ValueError): self._buf[start] = 0.0
        else:
            if isinstance(val, np.ndarray) and len(val) >= size:
                self._buf[start:start+size] = val[:size]
    if self.outputs['bus'] is not self._buf:
        self.outputs['bus'] = self._buf
```

The `is not self._buf` guard restores the ndarray reference after `Register.reset()` clears the output to the scalar `0` sentinel at timestep boundaries.

### 2. Single-op numpy fancy-index in BusSelector

`BusSelector` receives a `_flat_indices` array injected at compile time:

```python
# Compile time (once, in _build_bus_layout):
block._flat_indices = np.array([index_map[path] for path in block._paths], dtype=np.intp)

# Runtime (every FPI iteration):
def update(self, t=None):
    buf = self.inputs['bus']
    if not isinstance(buf, np.ndarray):
        return  # FPI zero-init sentinel
    if self._flat_indices is not None:
        self.outputs._data[:len(self.keys)] = buf[self._flat_indices]  # single op
        return
    # Fallback dict path (unit tests without sim.run()) — unchanged
    ...
```

`O(n_keys)` numpy slice replaces `O(n_keys × depth)` Python dict traversal per FPI call.

### 3. BusMerge: buffer concatenation at compile time

`BusMerge` receives a `_copy_plan` at compile time that specifies `(dst_start, dst_end, src_key)` slices. `update()` copies contiguous slices:

```python
def update(self, t=None):
    if self._copy_plan is not None:
        for dst_start, dst_end, src_key in self._copy_plan:
            src = self.inputs[src_key]
            if isinstance(src, np.ndarray):
                self._buf[dst_start:dst_end] = src
        if self.outputs['bus'] is not self._buf:
            self.outputs['bus'] = self._buf
        return
    # Fallback dict path — unchanged
    ...
```

### 4. Compile-time layout: `_build_bus_layout()`

`Simulation._build_bus_layout()` runs once at `Simulation.__init__` (via `_assemble_graph()`). It performs three passes over all blocks — strict producer-before-consumer ordering to handle blocks at the same DAG depth:

| Pass | Processes | Produces |
|------|-----------|---------|
| 1 | `BusCreator`, `BusFunction` | `_out_index_map` (key→flat index) |
| 2 | `BusMerge`, `Subsystem` | `_out_index_map`, `_copy_plan`, `_buf` |
| 3 | `BusSelector`, `BusFunction` | `_flat_indices`, `_in_flat_indices`, `_out_buf` |

This replaces the old `_check_bus_schemas()` call. Warnings for missing dot-path keys are emitted at `INFO` level into the simulation logger during Pass 3 (compile time, not per-timestep). `_check_bus_schemas()` is no longer called from `_run_loop()`.

The three-pass separation is necessary because non-algebraic (passthrough) Subsystems have `len() == 0`, placing them at DAG depth 0 alongside `BusCreator` blocks. Without pass separation, iteration order within a depth is non-deterministic and a `BusSelector` could process before its upstream `BusCreator`.

### 5. Anderson acceleration works on the flat buffer

`ConnectionBooster` detects bus connections via `current.dtype == object` (object-dtype register holding a float64 ndarray). Anderson acceleration then operates **directly on the flat float64 ndarray** — no helpers, no dict conversion:

```python
if current.dtype == object and len(current) > 0:
    bus_arr = current.flat[0]
    if not isinstance(bus_arr, np.ndarray):
        self.set(current); return float('inf')
    if self._bus_history is None:
        self._bus_history = bus_arr.copy()          # allocate once
        self._anderson_out = bus_arr.copy()
        self._anderson_wrapper = np.empty(1, dtype=object)
        self._anderson_wrapper[0] = self._anderson_out
        self.set(current); return float('inf')
    _val, res = self.accelerator.step(self._bus_history, bus_arr)
    np.copyto(self._bus_history, _val)
    np.copyto(self._anderson_out, _val)              # in-place, zero alloc
    self.set(self._anderson_wrapper)
    return res
```

After the first call allocates two fixed-size `float64` buffers, every subsequent FPI iteration performs **zero allocation**. Bus signals in algebraic loops converge super-linearly via Anderson (same as scalar signals), operating on the flat vector representation.

### 6. Benchmark results

Measured on Apple M-series (numpy 2.4.2, 200k iterations each). The "dict path" numbers reflect the best-case pre-refactor path with pre-split `_paths` tuples (no per-call string splitting).

| Operation | dict path (pre-refactor) | fast path (flat array) | Speedup |
|-----------|--------------------------|------------------------|---------|
| `BusSelector.update()` — 8-key bus, select 4 | ~1140 ns | ~380 ns | **3×** |
| `BusFunction.update()` — 8-key bus, 1 output | ~685 ns | ~725 ns | ~1× (func-call dominated) |

`BusSelector` scaling with bus width (fast path): ~370 ns regardless of width from 2 to 64 keys — **O(1) in bus width** because the entire extraction is a single numpy fancy-index slice.

Full simulation throughput (8-key bus, no algebraic loops, dt=0.01): **~40k timesteps/s**.

The `BusFunction` result is expected: the speedup from avoiding dict traversal is dominated by the Python function call overhead (`self.func(*vals)`). The fast path is architectural rather than a measurable gain for `BusFunction`.

---

## Integration with Existing PathSim

### What was **not** changed

- `_block.py`, `connection.py`, simulation core loop, all ODE solvers — untouched
- `Register` — unchanged; bus blocks declare `dtype=object` in `__init__`, an existing supported mode
- Graph construction, DAG ordering, event handling — untouched

### What was added/patched

**`simulation.py`** — new compile-time layout methods:
- `_build_bus_layout()` — three-pass compile step; replaces `_check_bus_schemas()`
- `_inject_bus_indices()` — injects `_flat_indices` into `BusSelector`/`BusFunction`, builds `_copy_plan` for `BusMerge`
- `_build_merge_layout()` — assembles merged index map and copy plan for `BusMerge`
- `_build_bus_layout_subsystem()` — recurses into `Subsystem` internals, crosses `Interface` boundaries
- `_get_subsystem_output_map()` — traces backward through a Subsystem's connections to find the upstream producer's index map (handles nested Subsystems and passthrough Interface)
- `_find_bus_input_map()` — traces backward through outer connections from a block's input port to find the upstream `_out_index_map`
- `_inject_scope_bus_info()` — names Scope channels after bus keys for PathView display
- `_inject_logger_into_blocks()` — sets `block._logger = sim.logger` so bus warnings route to PathView's log panel

**`optim/booster.py`** — simplified bus path in `ConnectionBooster.update()`:
- `_bus_history` / `_anderson_out` / `_anderson_wrapper` float64 buffers (lazy-allocated on first bus value)
- `current.dtype == object` detection (replaces old `_is_bus` tristate and `_flatten_bus_to_array` helpers)
- Anderson acceleration operates directly on the flat buffer (no dict round-trip)

---

## Bus Schemas (`Bus`/`BusElement`)

`BusCreator` accepts a `Bus` object or a plain list of strings. The `Bus` object is not required for runtime correctness but enables:
- Compile-time key validation and flat-index layout computation
- Schema-aware `Scope` recording (channels named by element name + unit)
- Clean `repr` output

```python
zone_bus = Bus('Zone', elements=[
    BusElement('Temperature', unit='C'),
    BusElement('Humidity',    unit='%RH'),
])

creator  = BusCreator(zone_bus)
selector = BusSelector(['Temperature'])      # valid — no warning
selector2 = BusSelector(['WindSpeed'])       # BUS WARNING logged at sim build time
```

Nested buses are expressed by setting `data_type=other_bus` on a `BusElement`. The layout builder recursively flattens nested schemas into dot-path keys (e.g., `'Zone.Temperature'`), and `BusSelector` dot-notation paths are pre-split into tuples at `__init__` time (`self._paths`) so no string work occurs in the hot path.

---

## What Was Intentionally Deferred

- **`BusElement.dimensions` enforcement** — stored as metadata, not validated against actual signal shape. The infrastructure is there but no block reads it yet.
- **`ScopeSignal` bus channel support** (F6) — for the `opt/` parameter estimation module. Depends on the param-est branch being merged first.

---

## Test Coverage

105 dedicated bus tests across two files:
- `tests/pathsim/blocks/test_bus.py` — 14 unit tests for each block in isolation (no `Simulation`)
- `tests/pathsim/test_bus.py` — 91 integration and schema tests (full `Simulation` runs, including Subsystem passthrough, BusFunction→BusMerge chains, algebraic loop convergence)

All 1088 non-OCP tests pass.

---

## Commit History (this branch, bus-related)

```
(flat-array refactor)
77bd635  bus: pre-compute dot-notation paths at init in BusSelector and BusFunction
0b5792c  bus: fix critical and important bugs from pre-merge review
056c066  bus: rename BusCreator keys→bus, remove plain/Bus branch from schema checker
d7e1936  bus: emit missing-key warnings at INFO level so PathView surfaces them
e24970b  bus: inject sim.logger into bus blocks so PathView log panel captures warnings
81b2d5b  bus: route warnings through logger, extend F5 schema check across Subsystem
6022823  bus: add BusFunction, F4/F5 fixes, bus-aware Scope, examples, docs
```
