#########################################################################################
##
##                               MAIN SIMULATION ENGINE
##                                   (simulation.py)
##
##                This module contains the simulation class that manages
##            the blocks, connections, events and specific simulation methods.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

import time
import datetime
import logging

from pathsim import __version__

from ._constants import (
    SIM_TIMESTEP,
    SIM_TIMESTEP_MIN,
    SIM_TIMESTEP_MAX,
    SIM_TOLERANCE_FPI,
    SIM_ITERATIONS_MAX,
    LOG_ENABLE
    )

from .optim.booster import ConnectionBooster

from .utils.graph import Graph
from .utils.analysis import Timer
from .utils.deprecation import deprecated
from .utils.portreference import PortReference
from .utils.progresstracker import ProgressTracker
from .utils.logger import LoggerManager

from .solvers import SSPRK22, SteadyState

from .blocks._block import Block

from .events._event import Event

from .connection import Connection


# MODULE HELPERS ========================================================================

def _bus_valid_paths(struct, prefix=''):
    """Yield every valid dot-path key in a ``Bus.structure_dict()`` result.

    Leaf entries (dicts containing a ``'data_type'`` key) are not recursed
    into; their path is yielded as-is.  Nested bus entries are recursed.
    """
    for key, val in struct.items():
        path = f'{prefix}.{key}' if prefix else key
        yield path
        if isinstance(val, dict) and 'data_type' not in val:
            yield from _bus_valid_paths(val, path)


# TRANSIENT SIMULATION CLASS ============================================================

class Simulation:
    """Class that performs transient analysis of the dynamical system, defined by the 
    blocks and connecions. It manages all the blocks and connections and the timestep update.

    The global system equation is evaluated by fixed point iteration, so the information from 
    each timestep gets distributed within the entire system and is available for all blocks at 
    all times.

    The minimum number of fixed-point iterations 'iterations_min' is set to 'None' by default 
    and then the length of the longest internal signal path (with passthrough) is used as the 
    estimate for minimum number of iterations needed for the information to reach all instant 
    time blocks in each timestep. Dont change this unless you know that the actual path is 
    shorter or something similar that prohibits instant time information flow. 

    Convergence check for the fixed-point iteration loop with 'tolerance_fpi' is based on 
    max absolute error (max-norm) to previous iteration and should not be touched.

    Multiple numerical integrators are implemented in the 'pathsim.solvers' module. 
    The default solver is a fixed timestep 2nd order Strong Stability Preserving Runge Kutta 
    (SSPRK22) method which is quite fast and has ok accuracy, especially if you are forced to 
    take small steps to cover the behaviour of forcing functions. Adaptive timestepping and 
    implicit integrators are also available.
    
    Manages an event handling system based on zero crossing detection. Uses 'Event' objects 
    to monitor solver states of stateful blocks and applys transformations on the state in 
    case an event is detected. 

    Example
    -------

    This is how to setup a simple system simulation using the 'Simulation' class:

    .. code-block:: python
        
        import numpy as np

        from pathsim import Simulation, Connection
        from pathsim.blocks import Source, Integrator, Scope

        src = Source(lambda t: np.cos(2*np.pi*t))
        itg = Integrator()
        sco = Scope(labels=["source", "integrator"])
        
        sim = Simulation(
            blocks=[src, itg, sco],
            connections=[
                Connection(src[0], itg[0], sco[0]),
                Connection(itg[0], sco[1])    
                ],
            dt=0.01
            )

        sim.run(4)
        sim.plot()

    Parameters
    ----------
    blocks : list[Block] 
        blocks that define the system
    connections : list[Connection] 
        connections that connect the blocks
    events : list[Event]
        list of event trackers (zero crossing detection, schedule, etc.)
    dt : float
        transient simulation timestep in time units, 
        default see ´SIM_TIMESTEP´ in ´_constants.py´
    dt_min : float
        lower bound for transient simulation timestep, 
        default see ´SIM_TIMESTEP_MIN´ in ´_constants.py´
    dt_max : float
        upper bound for transient simulation timestep, 
        default see ´SIM_TIMESTEP_MAX´ in ´_constants.py´
    Solver : Solver 
        ODE solver class for numerical integration from ´pathsim.solvers´,
        default is ´pathsim.solvers.ssprk22.SSPRK22´ (2nd order expl. Runge Kutta)
    tolerance_fpi : float
        absolute tolerance for convergence of algebraic loops 
        and internal optimizers of implicit ODE solvers, 
        default see ´SIM_TOLERANCE_FPI´ in ´_constants.py´
    iterations_max : int
        maximum allowed number of iterations for implicit ODE 
        solver optimizers and algebraic loop solver, 
        default see ´SIM_ITERATIONS_MAX´ in ´_constants.py´
    log : bool | string
        flag to enable logging, default see ´LOG_ENABLE´ in ´_constants.py´
        (alternatively a path to a log file can be specified)
    solver_kwargs : dict
        additional parameters for numerical solvers such as absolute 
        (´tolerance_lte_abs´) and relative (´tolerance_lte_rel´) tolerance, 
        defaults are defined in ´_constants.py´

    Attributes
    ----------
    time : float
        global simulation time, starting at ´0.0´
    graph : Graph
        internal graph representation for fast system funcion evluations 
        using DAG with algebraic depths
    boosters : None | list[ConnectionBooster]
        list of boosters (fixed point accelerators) that wrap algebraic 
        loop closing connections assembled from the system graph
    engine : Solver
        global integrator (ODE solver) instance serving as a dummy to 
        get attributes and access to intermediate evaluation stages
    logger : logging.Logger
        global simulation logger
    _blocks_dyn : set[Block]
        blocks with internal ´Solver´ instances (stateful) 
    _blocks_evt : set[Block]
        blocks with internal events (discrete time, eventful) 
    _active : bool
        flag for setting the simulation as active, used for interrupts
    """

    def __init__(
        self, 
        blocks=None, 
        connections=None, 
        events=None,
        dt=SIM_TIMESTEP, 
        dt_min=SIM_TIMESTEP_MIN, 
        dt_max=SIM_TIMESTEP_MAX, 
        Solver=SSPRK22, 
        tolerance_fpi=SIM_TOLERANCE_FPI, 
        iterations_max=SIM_ITERATIONS_MAX, 
        log=LOG_ENABLE,
        **solver_kwargs
        ):

        #system definition
        self.blocks      = set()
        self.connections = set()
        self.events      = set()

        #simulation timestep and bounds
        self.dt     = dt
        self.dt_min = dt_min
        self.dt_max = dt_max

        #numerical integrator to be used (class definition)
        self.Solver = Solver

        #numerical integrator instance
        self.engine = Solver()

        #internal system graph -> initialized later
        self.graph = None
        self._graph_dirty = False

        #internal algebraic loop solvers -> initialized later
        self.boosters = None

        #error tolerance for fixed point loop and implicit solver
        self.tolerance_fpi = tolerance_fpi

        #additional solver parameters
        self.solver_kwargs = solver_kwargs

        #iterations for fixed-point loop
        self.iterations_max = iterations_max

        #enable logging flag
        self.log = log

        #initial simulation time
        self.time = 0.0

        #collection of blocks with internal ODE solvers
        self._blocks_dyn = set()

        #collection of blocks with internal events
        self._blocks_evt = set()

        #flag for setting the simulation active
        self._active = True

        #initialize logging 
        logger_mgr = LoggerManager(
            enabled=bool(self.log),
            output=self.log if isinstance(self.log, str) else None,
            level=logging.INFO,
            date_format='%H:%M:%S'
            )
        self.logger = logger_mgr.get_logger("simulation")
        self.logger.info(f"LOGGING (log: {self.log})")

        #prepare and add blocks (including internal events)
        if blocks is not None:
            for block in blocks:
                self.add_block(block)

        #check and add connections
        if connections is not None:
            for connection in connections:
                self.add_connection(connection)

        #check and add events
        if events is not None:
            for event in events:
                self.add_event(event)

        #check if blocks from connections are in simulation
        self._check_blocks_are_managed()

        #assemble the system graph for simulation
        self._assemble_graph()


    def __contains__(self, other):
        """Check if blocks, connections or events are 
        already part of the simulation 

        Paramters
        ---------
        other : obj
            object to check if its part of simulation

        Returns
        -------
        bool
        """
        return (
            other in self.blocks or 
            other in self.connections or 
            other in self.events
            )


    def __bool__(self):
        """Boolean evaluation of Simulation instances

        Returns
        -------
        active : bool
            is the simulation active
        """
        return self._active


    # methods for access to metadata ----------------------------------------------

    @property
    def size(self):
        """Get size information of the simulation, such as total number 
        of blocks and dynamic states, with recursive retrieval from subsystems

        Returns
        -------
        sizes : tuple[int]
            size of simulation (number of blocks) and number 
            of internal states (from internal engines)
        """
        total_n, total_nx = 0, 0
        for block in self.blocks:
            n, nx = block.size
            total_n += n
            total_nx += nx
        return total_n, total_nx


    # visualization ---------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Plot the simulation results by calling all the blocks 
        that have visualization capabilities such as the 'Scope' 
        and 'Spectrum'.

        This is a quality of life method. Blocks can be visualized 
        individually due to the object oriented nature, but it might 
        be nice to just call the plot metho globally and look at all 
        the results at once. Also works for models loaded from an 
        external file.

        Parameters
        ----------
        args : tuple
            args for the plot methods
        kwargs : dict
            kwargs for the plot method
        """
        for block in self.blocks:
            if block: block.plot(*args, **kwargs)


    # adding system components ----------------------------------------------------

    def add_block(self, block):
        """Adds a new block to the simulation, initializes its local solver
        instance and collects internal events of the new block.

        This works dynamically for running simulations.

        Parameters
        ----------
        block : Block
            block to add to the simulation
        """

        #check if block already in block list
        if block in self.blocks:
            _msg = f"block {block} already part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #initialize numerical integrator of block with parent
        block.set_solver(self.Solver, self.engine, **self.solver_kwargs)

        #add to dynamic list if solver was initialized
        if block.engine:
            self._blocks_dyn.add(block)

        #add to eventful list if internal events
        if block.events:
            self._blocks_evt.add(block)

        #add block to global blocklist
        self.blocks.add(block)

        #mark graph for rebuild
        if self.graph:
            self._graph_dirty = True


    def remove_block(self, block):
        """Removes a block from the simulation.

        This works dynamically for running simulations. The graph
        is lazily rebuilt on the next simulation update.

        Parameters
        ----------
        block : Block
            block to remove from the simulation
        """

        #check if block is in block list
        if block not in self.blocks:
            _msg = f"block {block} not part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #remove from global blocklist
        self.blocks.discard(block)

        #remove from dynamic list
        self._blocks_dyn.discard(block)

        #remove from eventful list
        self._blocks_evt.discard(block)

        #mark graph for rebuild
        if self.graph:
            self._graph_dirty = True


    def add_connection(self, connection):
        """Adds a new connection to the simulation and checks if
        the new connection overwrites any existing connections.

        This works dynamically for running simulations.

        Parameters
        ----------
        connection : Connection
            connection to add to the simulation
        """

        #check if connection already in connection list
        if connection in self.connections:
            _msg = f"{connection} already part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #add connection to global connection list
        self.connections.add(connection)

        #mark graph for rebuild
        if self.graph:
            self._graph_dirty = True


    def remove_connection(self, connection):
        """Removes a connection from the simulation.

        This works dynamically for running simulations. The graph
        is lazily rebuilt on the next simulation update.

        Parameters
        ----------
        connection : Connection
            connection to remove from the simulation
        """

        #check if connection is in connection list
        if connection not in self.connections:
            _msg = f"{connection} not part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #remove from global connection list
        self.connections.discard(connection)

        #mark graph for rebuild
        if self.graph:
            self._graph_dirty = True


    def add_event(self, event):
        """Checks and adds a new event to the simulation.

        This works dynamically for running simulations.

        Parameters
        ----------
        event : Event
            event to add to the simulation
        """

        #check if event already in event list
        if event in self.events:
            _msg = f"{event} already part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #add event to global event list
        self.events.add(event)


    def remove_event(self, event):
        """Removes an event from the simulation.

        This works dynamically for running simulations.

        Parameters
        ----------
        event : Event
            event to remove from the simulation
        """

        #check if event is in event list
        if event not in self.events:
            _msg = f"{event} not part of simulation"
            self.logger.error(_msg)
            raise ValueError(_msg)

        #remove from global event list
        self.events.discard(event)


    # system assembly -------------------------------------------------------------

    def _check_bus_schemas(self, connections=None):
        """Warn when a BusSelector receives keys absent from the upstream BusCreator schema.

        Traces bus signals forward from every BusCreator in *connections*,
        crossing Subsystem boundaries (including pure passthrough wires), and
        warns for each BusSelector that requests a key not present in the schema.

        Parameters
        ----------
        connections : iterable or None
            Connection set to check.  Defaults to ``self.connections`` (outer
            level).  Passed recursively to check BusCreators inside Subsystems.
        """
        from .blocks.buses import BusCreator
        from .subsystem import Subsystem

        if connections is None:
            connections = self.connections

        seen_subsystems = set()

        for conn in connections:
            src = conn.source.block

            # Collect subsystems at this level for recursive descent.
            if isinstance(src, Subsystem):
                seen_subsystems.add(src)
            for trg_ref in conn.targets:
                if isinstance(trg_ref.block, Subsystem):
                    seen_subsystems.add(trg_ref.block)

            if not isinstance(src, BusCreator):
                continue

            # Build valid schema for this BusCreator.
            valid = set(_bus_valid_paths(src.bus.structure_dict()))

            for trg_ref in conn.targets:
                self._trace_bus_forward(
                    trg_ref.block, trg_ref.ports[0], valid, src,
                    current_conns=connections,
                )

        # Recurse into every Subsystem seen at this level.
        for sub in seen_subsystems:
            self._check_bus_schemas(sub.connections)


    def _trace_bus_forward(self, block, in_port, valid, origin,
                           current_conns, conns_stack=None, _depth=0):
        """Follow a bus signal forward and warn for BusSelectors with missing keys.

        Parameters
        ----------
        block : Block
            Block receiving the bus signal.
        in_port : int
            Input port index on *block*.
        valid : set[str]
            Valid dot-path keys for this bus schema.
        origin : BusCreator
            The originating BusCreator (used in warning messages).
        current_conns : iterable
            Connections at the level where *block* lives.
        conns_stack : list[(outer_conns, parent_subsystem)] or None
            Stack of outer-level context, used to exit Subsystem boundaries.
        _depth : int
            Recursion guard — stops at 20 levels.
        """
        if _depth > 20:
            return
        if conns_stack is None:
            conns_stack = []

        from .blocks.buses import BusSelector
        from .subsystem import Subsystem, Interface

        if isinstance(block, BusSelector):
            missing = [k for k in block.keys if k not in valid]
            if missing:
                self.logger.info(
                    f"BUS WARNING: {block!r} requests key(s) {missing!r} "
                    f"not in {origin!r} schema. Output will be 0.0."
                )
            return

        from .blocks.buses import BusFunction
        if isinstance(block, BusFunction):
            # BusFunction transforms the schema — output keys are known statically.
            new_valid = set(block.out_keys)
            for out_conn in current_conns:
                if out_conn.source.block is block and out_conn.source.ports[0] == 0:
                    for trg in out_conn.targets:
                        self._trace_bus_forward(
                            trg.block, trg.ports[0], new_valid, block,
                            current_conns=current_conns,
                            conns_stack=conns_stack,
                            _depth=_depth + 1,
                        )
            return

        if isinstance(block, Subsystem):
            # Bus enters subsystem at port in_port (= Interface.outputs[in_port]).
            iface = block.interface
            new_stack = conns_stack + [(current_conns, block)]
            for inner_conn in block.connections:
                if (inner_conn.source.block is iface
                        and inner_conn.source.ports[0] == in_port):
                    for inner_trg in inner_conn.targets:
                        self._trace_bus_forward(
                            inner_trg.block, inner_trg.ports[0], valid, origin,
                            current_conns=block.connections,
                            conns_stack=new_stack,
                            _depth=_depth + 1,
                        )
            return

        if isinstance(block, Interface):
            # Bus is exiting a Subsystem via Interface.inputs[in_port]
            # (= parent Subsystem outputs[in_port]).  Pop the stack and follow
            # outer connections from the parent Subsystem at that output port.
            if conns_stack:
                outer_conns, parent_sub = conns_stack[-1]
                remaining_stack = conns_stack[:-1]
                for outer_conn in outer_conns:
                    if (outer_conn.source.block is parent_sub
                            and outer_conn.source.ports[0] == in_port):
                        for outer_trg in outer_conn.targets:
                            self._trace_bus_forward(
                                outer_trg.block, outer_trg.ports[0], valid, origin,
                                current_conns=outer_conns,
                                conns_stack=remaining_stack,
                                _depth=_depth + 1,
                            )
            return


    def _assemble_graph(self):
        """Build the internal graph representation for fast system function
        evaluation and algebraic loop resolution.
        """

        #reset all block inputs to clear stale values from removed connections
        for block in self.blocks:
            block.inputs.reset()

        #time the graph construction
        with Timer(verbose=False) as T:
            self.graph = Graph(self.blocks, self.connections)
        self._graph_dirty = False

        #create boosters for loop closing connections
        if self.graph.has_loops:
            self.boosters = [
                ConnectionBooster(conn) for conn in self.graph.loop_closing_connections()
            ]

        #log block summary
        num_dynamic = len(self._blocks_dyn)
        num_static = len(self.blocks) - num_dynamic
        num_eventful = len(self._blocks_evt)
        self.logger.info(
            f"BLOCKS (total: {len(self.blocks)}, dynamic: {num_dynamic}, "
            f"static: {num_static}, eventful: {num_eventful})"
            )

        #log graph info
        self.logger.info(
            "GRAPH (nodes: {}, edges: {}, alg. depth: {}, loop depth: {}, runtime: {})".format(
                *self.graph.size, *self.graph.depth, T
                )
            )

        #inject simulation logger into bus blocks so PathView's log panel sees them
        self._inject_logger_into_blocks(self.blocks)

        #compile bus topology: pre-compute flat indices and buffers for bus blocks
        self._build_bus_layout()


    def _inject_logger_into_blocks(self, blocks):
        """Set _logger on any block (or nested Subsystem block) that declares it.

        Bus blocks (BusSelector, BusFunction, BusMerge) default their _logger to
        the module-level pathsim.blocks.buses logger.  By replacing it with
        self.logger (pathsim.simulation), their runtime warnings flow through the
        same logger that PathView hooks into.
        """
        from .subsystem import Subsystem
        for block in blocks:
            if hasattr(block, '_logger'):
                block._logger = self.logger
            if isinstance(block, Subsystem):
                self._inject_logger_into_blocks(block.blocks)


    def _build_bus_layout(self):
        """Compile bus topology: pre-compute flat indices and buffers.

        This is the bus "compile" step, called once per graph assembly.  It walks the
        system topology in topological order (using the DAG), and for every bus-producing
        block (``BusCreator``, ``BusFunction``, ``BusMerge``) it:

        - Stores an ``_out_index_map`` (``{dot_path: flat_int_index}``) on the block.
        - Injects ``_flat_indices`` into downstream ``BusSelector`` blocks.
        - Injects ``_in_flat_indices`` + ``_out_buf`` into downstream ``BusFunction`` blocks.
        - Builds ``_copy_plan`` + ``_buf`` for downstream ``BusMerge`` blocks.
        - Injects ``_bus_ports`` and labels into ``Scope`` blocks directly wired to a bus.

        Subsystem boundaries are handled via ``_build_bus_layout_subsystem``.
        """
        from .blocks.buses import (
            BusCreator, BusSelector, BusFunction, BusMerge,
            _BusProducer, _BusMerger, _BusConsumer,
            _bus_key_index_map, _bus_leaf_count,
        )
        from .blocks.scope import Scope
        from .subsystem import Subsystem

        # Reset build-time attributes on all bus blocks so a re-assemble starts clean.
        for block in self.blocks:
            if isinstance(block, (_BusProducer, _BusMerger)):
                block._out_index_map = None
            if isinstance(block, BusSelector):
                block._flat_indices = None
            if isinstance(block, BusFunction):
                block._in_flat_indices = None
                block._out_buf = None
            if isinstance(block, _BusMerger):
                block._copy_plan = None
                block._buf = None
            if isinstance(block, Subsystem):
                self._reset_bus_layout_subsystem(block)

        def _resolve_indices(keys_paths, in_map, block, context=''):
            """Resolve dot-paths against *in_map*; warn once per missing key."""
            indices = []
            for path in keys_paths:
                dot = '.'.join(path)
                if dot in in_map:
                    indices.append(in_map[dot])
                else:
                    self.logger.info(
                        "BUS WARNING: %r key %r not found in upstream bus schema%s. "
                        "Output will be 0.0.",
                        block, dot, f" ({context})" if context else '',
                    )
                    indices.append(0)
            return indices

        def _process_block(block):
            """Process a single block in the bus layout build."""
            if isinstance(block, BusCreator):
                m, _ = _bus_key_index_map(block.bus)
                block._out_index_map = m

            elif isinstance(block, BusFunction):
                in_map = self._find_bus_input_map(
                    block, block.input_port_labels['bus'], self.connections
                )
                if in_map is not None:
                    indices = _resolve_indices(
                        block._in_paths, in_map, block, 'in_keys'
                    )
                    block._in_flat_indices = np.array(indices, dtype=np.intp)
                    block._out_buf = np.zeros(len(block.out_keys))
                # Output map is always just the out_keys in order
                block._out_index_map = {k: i for i, k in enumerate(block.out_keys)}

            elif isinstance(block, BusMerge):
                self._build_merge_layout(block, self.connections)

            elif isinstance(block, BusSelector):
                in_map = self._find_bus_input_map(
                    block, block.input_port_labels['bus'], self.connections
                )
                if in_map is not None:
                    indices = _resolve_indices(block._paths, in_map, block)
                    block._flat_indices = np.array(indices, dtype=np.intp)

            elif isinstance(block, Subsystem):
                self._build_bus_layout_subsystem(block, self.connections)

        # Three-pass layout build:
        #
        # The fundamental ordering challenge: a non-algebraic (len==0) passthrough
        # Subsystem between a BusCreator and a BusSelector puts BOTH at the same
        # DAG depth.  If the BusSelector is iterated first its upstream _out_index_map
        # is None (BusCreator not yet visited).
        #
        # Solution — strict producer-before-consumer ordering, independent of DAG depth:
        #
        #   Pass 1: BusCreator, BusFunction
        #       → set _out_index_map (no upstream deps; BusFunction's is just {k:i}).
        #   Pass 2: BusMerge, Subsystem
        #       → set _out_index_map (needs Pass-1 maps; BusMerge merges upstream maps).
        #   Pass 3: BusSelector, BusFunction
        #       → set _flat_indices / _in_flat_indices (all upstream maps now populated).
        #
        # Fourth pass handles Scope (len==0 → placed at DAG depth 0 before all producers).

        all_blocks = []
        for _, blocks, _ in self.graph.dag():
            all_blocks.extend(blocks)
        for _, blocks, _ in self.graph.loop():
            all_blocks.extend(blocks)

        for block in all_blocks:
            if isinstance(block, _BusProducer):       # BusCreator, BusFunction
                _process_block(block)

        for block in all_blocks:
            if isinstance(block, (_BusMerger, Subsystem)):  # BusMerge, nested Subsystems
                _process_block(block)

        for block in all_blocks:
            if isinstance(block, _BusConsumer):       # BusSelector, BusFunction
                _process_block(block)

        # Fourth pass: inject Scope bus info after all producers have _out_index_map.
        for block in self.blocks:
            if isinstance(block, Scope):
                self._inject_scope_bus_info(block, self.connections)


    def _reset_bus_layout_subsystem(self, sub):
        """Recursively reset bus layout attributes inside a Subsystem."""
        from .blocks.buses import (
            BusSelector, BusFunction,
            _BusProducer, _BusMerger,
        )
        from .blocks.scope import Scope
        from .subsystem import Subsystem
        for block in sub.blocks:
            if isinstance(block, (_BusProducer, _BusMerger)):
                block._out_index_map = None
            if isinstance(block, BusSelector):
                block._flat_indices = None
            if isinstance(block, BusFunction):
                block._in_flat_indices = None
                block._out_buf = None
            if isinstance(block, _BusMerger):
                block._copy_plan = None
                block._buf = None
            if isinstance(block, Subsystem):
                self._reset_bus_layout_subsystem(block)


    def _find_bus_input_map(self, block, in_port_idx, connections):
        """Return the ``_out_index_map`` of the block feeding *block* at *in_port_idx*.

        Looks through *connections* for a connection whose target is *block* at
        *in_port_idx*, then returns ``getattr(source, '_out_index_map', None)``.
        Returns ``None`` if no such connection exists or the source has no map.

        For ``Subsystem`` sources the map is looked up via the Subsystem's internal
        connections (the Interface → inner bus producer chain).
        """
        from .subsystem import Subsystem
        for conn in connections:
            for trg in conn.targets:
                # PortReference.ports stores the raw port specifier (str or int);
                # resolve through the block's input mapping to get an integer index.
                resolved = trg.block.inputs._map(trg.ports[0])
                if trg.block is block and resolved == in_port_idx:
                    src = conn.source.block
                    src_port = conn.source.ports[0]
                    if isinstance(src, Subsystem):
                        return self._get_subsystem_output_map(src, src_port)
                    return getattr(src, '_out_index_map', None)
        return None


    def _get_subsystem_output_map(self, sub, out_port):
        """Find the ``_out_index_map`` that exits a Subsystem at *out_port*.

        Traces backward from Interface.inputs[out_port] to the inner producer.
        Also recursively processes bus blocks inside the Subsystem if not yet done.
        """
        from .subsystem import Subsystem, Interface
        iface = sub.interface
        for inner_conn in sub.connections:
            for trg in inner_conn.targets:
                if trg.block is iface and trg.ports[0] == out_port:
                    src = inner_conn.source.block
                    src_port = inner_conn.source.ports[0]
                    if isinstance(src, Subsystem):
                        return self._get_subsystem_output_map(src, src_port)
                    if isinstance(src, Interface):
                        # Passthrough: the source IS the Interface (external input).
                        # Trace back through outer connections to the upstream producer.
                        return self._find_bus_input_map(sub, src_port, self.connections)
                    m = getattr(src, '_out_index_map', None)
                    if m is None:
                        # May need to build layout inside subsystem first
                        self._build_bus_layout_subsystem(sub, self.connections)
                        m = getattr(src, '_out_index_map', None)
                    return m
        return None


    def _build_merge_layout(self, merge_block, connections):
        """Build the copy plan and output buffer for a BusMerge block.

        Scans *connections* for all inputs to *merge_block*, collects their
        ``_out_index_map``\\s, concatenates them, and stores the result on the block.
        """
        input_maps = {}
        for i in range(merge_block.n):
            port_key = f'bus_{i}'
            port_idx = merge_block.input_port_labels[port_key]
            src_map = self._find_bus_input_map(merge_block, port_idx, connections)
            if src_map is not None:
                input_maps[i] = src_map

        if not input_maps:
            return  # No bus inputs found — keep dict fallback

        merged_map = {}
        copy_plan = []
        offset = 0
        for i in range(merge_block.n):
            if i not in input_maps:
                continue
            src_map = input_maps[i]
            # Size = number of leaf slots = max_index + 1
            size = max(src_map.values()) + 1 if src_map else 0
            port_key = f'bus_{i}'
            copy_plan.append((offset, offset + size, port_key))
            for path, idx in src_map.items():
                shifted_idx = offset + idx
                if path not in merged_map:
                    merged_map[path] = shifted_idx
                elif merge_block.on_conflict in ('last', 'warn'):
                    merged_map[path] = shifted_idx
                # 'first': keep existing entry (do nothing)
                # 'error': not handled at compile time (runtime dict-path raises)
            offset += size

        merge_block._buf = np.zeros(offset)
        merge_block._copy_plan = copy_plan
        merge_block._out_index_map = merged_map


    def _inject_scope_bus_info(self, scope, connections):
        """Inject bus layout info into a Scope that is directly wired to a bus signal.

        For each input port of *scope* that is driven by a bus-producing block,
        sets ``scope._bus_ports[port_index]`` to the ordered list of dot-path keys
        and (if no labels were supplied by the user) populates ``scope.labels``.
        """
        from .blocks.scope import Scope
        if not isinstance(scope, Scope):
            return

        for conn in connections:
            for trg in conn.targets:
                if trg.block is not scope:
                    continue
                port_idx = trg.ports[0]
                src = conn.source.block
                src_port = conn.source.ports[0]
                from .subsystem import Subsystem
                if isinstance(src, Subsystem):
                    in_map = self._get_subsystem_output_map(src, src_port)
                else:
                    in_map = getattr(src, '_out_index_map', None)
                if in_map is None:
                    continue

                # Ordered paths sorted by flat index
                ordered_paths = sorted(in_map.keys(), key=lambda p: in_map[p])

                if scope._bus_ports is None:
                    scope._bus_ports = {}
                scope._bus_ports[port_idx] = ordered_paths

                # Populate labels from bus paths if not user-supplied
                if not scope.labels:
                    scope.labels = list(ordered_paths)


    def _build_bus_layout_subsystem(self, sub, outer_connections):
        """Recursively build bus layout for blocks inside *sub*.

        Processes bus-producing blocks inside the Subsystem in connection order
        (no full topological sort — iterates forward over sub.connections).
        Also exports ``_out_index_map`` on blocks that feed out through Interface.
        """
        from .blocks.buses import (
            BusCreator, BusSelector, BusFunction,
            _BusMerger, _bus_key_index_map,
        )
        from .blocks.scope import Scope
        from .subsystem import Subsystem, Interface

        # Process inner bus producers (order: follow connections)
        for inner_conn in sub.connections:
            src = inner_conn.source.block

            if isinstance(src, BusCreator) and src._out_index_map is None:
                m, _ = _bus_key_index_map(src.bus)
                src._out_index_map = m

            elif isinstance(src, BusFunction) and src._out_index_map is None:
                in_map = self._find_bus_input_map(
                    src, src.input_port_labels['bus'], sub.connections
                )
                if in_map is not None:
                    indices = []
                    for p in src._in_paths:
                        dot = '.'.join(p)
                        if dot in in_map:
                            indices.append(in_map[dot])
                        else:
                            self.logger.info(
                                "BUS WARNING: %r in_key %r not found in upstream "
                                "bus schema. Output will be 0.0.", src, dot,
                            )
                            indices.append(0)
                    src._in_flat_indices = np.array(indices, dtype=np.intp)
                    src._out_buf = np.zeros(len(src.out_keys))
                src._out_index_map = {k: i for i, k in enumerate(src.out_keys)}

            elif isinstance(src, _BusMerger) and src._out_index_map is None:
                self._build_merge_layout(src, sub.connections)

            elif isinstance(src, Subsystem):
                self._build_bus_layout_subsystem(src, sub.connections)

            # Inject into inner consumers
            for trg_ref in inner_conn.targets:
                trg = trg_ref.block
                if isinstance(trg, BusSelector) and trg._flat_indices is None:
                    in_map = self._find_bus_input_map(
                        trg, trg.input_port_labels['bus'], sub.connections
                    )
                    # If source is Interface (= external input to subsystem), trace
                    # back through outer_connections to find the upstream producer.
                    if in_map is None and isinstance(inner_conn.source.block, Interface):
                        src_port = inner_conn.source.ports[0]
                        in_map = self._find_bus_input_map(sub, src_port, outer_connections)
                    if in_map is not None:
                        indices = []
                        for p in trg._paths:
                            dot = '.'.join(p)
                            if dot in in_map:
                                indices.append(in_map[dot])
                            else:
                                self.logger.info(
                                    "BUS WARNING: %r key %r not found in upstream "
                                    "bus schema. Output will be 0.0.", trg, dot,
                                )
                                indices.append(0)
                        trg._flat_indices = np.array(indices, dtype=np.intp)
                elif isinstance(trg, Scope):
                    self._inject_scope_bus_info(trg, sub.connections)


    # topological checks ----------------------------------------------------------

    def _check_blocks_are_managed(self):
        """Check whether the blocks that are part of the connections are 
        in the simulation block set ('self.blocks') and therefore managed 
        by the simulation.

        If not, there will be a warning in the logging.            
        """

        # Collect connection blocks
        conn_blocks = set()
        for conn in self.connections:
            conn_blocks.update(conn.get_blocks())

        # Check subset actively managed
        if not conn_blocks.issubset(self.blocks):
            self.logger.warning(
                f"{blk} in 'connections' but not in 'blocks'!"
                )


    # solver management -----------------------------------------------------------

    def _set_solver(self, Solver=None, **solver_kwargs):
        """Initialize all blocks with solver for numerical integration
        and tolerance for local truncation error ´tolerance_lte´.

        If blocks already have solvers, change the numerical integrator
        to the ´Solver´ class.

        Parameters
        ----------
        Solver : Solver
            numerical solver definition from ´pathsim.solvers´
        solver_kwargs : dict
            additional parameters for numerical solvers
        """

        #update global solver class
        if Solver is not None:
            self.Solver = Solver

        #update solver parmeters
        self.solver_kwargs.update(solver_kwargs)

        #initialize dummy engine to get solver attributes
        self.engine = self.Solver()

        #iterate all blocks and set integration engines with tolerances
        self._blocks_dyn = set()
        for block in self.blocks:
            block.set_solver(self.Solver, self.engine, **self.solver_kwargs)
            
            #add dynamic blocks to list
            if block.engine:
                self._blocks_dyn.add(block)

        #logging message
        self.logger.info(
            "SOLVER (dyn. blocks: {}) -> {} (adaptive: {}, explicit: {})".format(
                len(self._blocks_dyn),
                self.engine,
                self.engine.is_adaptive,
                self.engine.is_explicit
                )
            )


    # resetting -------------------------------------------------------------------

    def reset(self, time=0.0):
        """Reset the blocks to their initial state and the global time of 
        the simulation. 

        For recording blocks such as 'Scope', their recorded 
        data is also reset. 

        Resets linearization automatically, since resetting the blocks 
        resets their internal operators.

        Afterwards the system function is evaluated with '_update' to update
        the block inputs and outputs.

        Parameters
        ----------
        time : float
            simulation time for reset
        """

        self.logger.info(f"RESET (time: {time})")

        #set active again
        self._active = True

        #reset simulation time
        self.time = time

        #reset integration engine
        self.engine.reset()

        #reset all blocks to initial state
        for block in self.blocks:
            block.reset()

        #reset all event managers
        for event in self.events:
            event.reset()

        #evaluate system function
        self._update(self.time)


    # linearization ---------------------------------------------------------------

    def linearize(self):
        """Linearize the full system in the current simulation state 
        at the current simulation time.
        
        This is achieved by linearizing algebraic and dynamic operators 
        of the internal blocks. See definition of the 'Block' class.
    
        Before linearization, the global system function is evaluated 
        to get the blocks into the current simulation state. 
        This is only really relevant if no solving attempt has been 
        happened before.
        """
        #evaluate system function at current time
        self._update(self.time)

        #linearize all internal blocks and time it
        with Timer(verbose=False) as T:
            for block in self.blocks:
                block.linearize(self.time)

        self.logger.info(f"LINEARIZED (runtime: {T})")


    def delinearize(self):
        """Revert the linearization of the full system."""
        for block in self.blocks: 
            block.delinearize()

        self.logger.info("DELINEARIZED")


    # event system helpers --------------------------------------------------------

    def _get_active_events(self):
        """Generator that yields all active events from simulation
        and internal block events.
        """
        for event in self.events:
            if event:
                yield event
        for block in self._blocks_evt:
            for event in block.events:
                if event:
                    yield event


    def _estimate_events(self, t):
        """Estimate the time until the next.

        Parameters
        ----------
        t : float 
            evaluation time for event estimation

        Returns
        -------
        float | None
            esimated time until next event (delta)
        """

        dt_evt_min = None

        #check external events
        for event in self._get_active_events():

            #get the estimate
            dt_evt = event.estimate(self.time)

            #no estimate available
            if dt_evt is None: continue
            
            #smaller than min
            if dt_evt_min is None or dt_evt < dt_evt_min:
                dt_evt_min = dt_evt

        #return time until next event or None
        return dt_evt_min


    def _detected_events(self, t):
        """Check for possible (active) events and return them chronologically, 
        sorted by their timestep ratios (closest to the initial point in time).
    
        Parameters
        ----------
        t : float
            evaluation time for event function

        Returns
        -------
        detected : list[Event]
            list of detected events within timestep
        """

        #iterate all event managers
        detected_events = []
        for event in self._get_active_events():
            
            #check if an event is detected
            detected, close, ratio = event.detect(t)

            #event was detected during the timestep 
            if detected:
                detected_events.append([event, close, ratio])

        #return detected events sorted by ratio
        return sorted(detected_events, key=lambda e: e[-1])


    # solving system equations ----------------------------------------------------

    def _update(self, t):        
        """Distribute information within the system by evaluating the directed acyclic graph 
        (DAG) formed by the algebraic passthroughs of the blocks and resolving algebraic loops 
        through accelerated fixed-point iterations.
        
        Effectively evaluates the right hand side function of the global 
        system ODE/DAE

        .. math:: 
    
            \\begin{equnarray}
                \\dot{x} &= f(x, t) \\\\
                       0 &= g(x, t) 
            \\end{equnarray}

        by converging the whole system (´f´ and ´g´) to a fixed-point at a given point 
        in time ´t´.

        If no algebraic loops are present in the system, convergence is 
        guaranteed after the first stage (evaluation of the DAG in '_dag'). 

        Otherwise, accelerated fixed-point iterations ('_loops') are performed as a second 
        stage on the DAGs (broken cycles) of blocks that are part of or tainted by upstream 
        algebraic loops. 

        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #lazy graph rebuild if dirty
        if self._graph_dirty:
            self._assemble_graph()
            self._graph_dirty = False

        #evaluate DAG
        self._dag(t)

        #algebraic loops -> solve them
        if self.graph.has_loops:
            self._loops(t)


    def _dag(self, t):
        """Update the directed acyclic graph components of the system.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #perform gauss-seidel iterations without error checking
        for _, blocks_dag, connections_dag in self.graph.dag():

            #update blocks at algebraic depth (no error control)
            for block in blocks_dag:
                if block: block.update(t)

            #update connenctions at algebraic depth (data transfer)
            for connection in connections_dag:
                if connection: connection.update()


    def _loops(self, t):
        """Perform the algebraic loop solve of the system using accelerated 
        fixed-point iterations on the broken loop directed graph.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #reset accelerators of loop closing connections
        for con_booster in self.boosters:
            con_booster.reset()

        #perform solver iterations on algebraic loops
        for iteration in range(1, self.iterations_max):
            
            #iterate DAG depths of broken loops
            for _, blocks_loop, connections_loop in self.graph.loop():

                #update blocks at algebraic depth
                for block in blocks_loop:
                    if block: block.update(t)

                #update connenctions at algebraic depth (data transfer)
                for connection in connections_loop:
                    if connection: connection.update()

            #step boosters of loop closing connections
            max_err = 0.0
            for con_booster in self.boosters:
                err = con_booster.update()
                if err > max_err:
                    max_err = err
                       
            #check convergence
            if max_err <= self.tolerance_fpi:
                return

        #not converged -> error
        _msg = "algebraic loop not converged (iters: {}, err: {})".format(
            self.iterations_max, max_err
            )
        self.logger.error(_msg)
        raise RuntimeError(_msg)


    def _solve(self, t, dt):
        """For implicit solvers, this method implements the solving step 
        of the implicit update equation.

        It already involves the evaluation of the system equation with 
        the '_update' method within the loop.

        This also tracks the evolution of the solution as an estimate 
        for the convergence via the max residual norm of the fixed point 
        equation of the previous solution.

        Parameters
        ----------
        t : float
            evaluation time for system function
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #total evaluations of system equation
        total_evals = 0

        #perform fixed-point iterations to solve implicit update equation
        for it in range(self.iterations_max):

            #evaluate system equation (this is a fixed point loop)
            self._update(t)
            total_evals += 1            

            #advance solution of implicit solver
            max_error = 0.0
            for block in self._blocks_dyn:

                #skip inactive blocks
                if not block: 
                    continue
                
                #advance solution (internal optimizer)
                error = block.solve(t, dt)
                if error > max_error:
                    max_error = error

            #check for convergence (only error)
            if max_error <= self.tolerance_fpi:
                return True, total_evals, it+1

        #not converged in 'self.iterations_max' steps
        return False, total_evals, self.iterations_max


    def steadystate(self, reset=False): 
        """Find steady state solution (DC operating point) of the system 
        by switching all blocks to steady state solver, solving the 
        fixed point equations, then switching back.

        The steady state solver forces all the temporal derivatives, i.e.
        the right hand side equation (including external inputs) of the 
        engines of dynamic blocks to zero.

        Note
        ----
        This is really a sort of pseudo-steady-state solve. It does NOT compute 
        the limit :math:`t\\rightarrow\\infty` but rather forces all time 
        derivatives to zero at a given moment in time. 

        This means, for a given `t` it computes the block states `x` such that:
    
        .. math:: 
    
            0 = f(x, t)

        instead of the real steady state:

        .. math:: 

            \\lim_{t \\rightarrow \\infty} x(t)
        
            
        Parameters
        ----------
        reset : bool
            reset the simulation before solving for steady state (default False)
        """

        #reset the simulation before solving
        if reset:
            self.reset()

        #current solver class
        _solver = self.Solver
        
        #switch to steady state solver
        self._set_solver(SteadyState)

        #log message begin of steady state solver
        self.logger.info(f"STEADYSTATE -> STARTING (reset: {reset})")

        #solve for steady state at current time
        with Timer(verbose=False) as T:
            success, evals, iters = self._solve(self.time, self.dt)

        #catch non convergence
        if not success:
            _msg = "STEADYSTATE -> FINISHED (success: {}, evals: {}, iters: {}, runtime: {})".format(
                success, evals, iters, T)
            self.logger.error(_msg)
            raise RuntimeError(_msg)

        #sample result
        self._sample(self.time, self.dt)

        #log message
        self.logger.info(
            "STEADYSTATE -> FINISHED (success: {}, evals: {}, iters: {}, runtime: {})".format(
                success, evals, iters, T)
            )

        #switch back to original solver
        self._set_solver(_solver)


    # timestepping helpers --------------------------------------------------------

    def _revert(self, t):
        """Revert simulation state to previous timestep for adaptive solvers 
        when local truncation error is too large and timestep has to be 
        retaken with smaller timestep.

        Parameters
        ----------
        t : float
            evaluation time for simulation revert 
        """

        #revert dummy engine (for history, allways)
        self.engine.revert()

        #revert block states
        for block in self._blocks_dyn:
            if block: block.revert()

        #update the simulation (evaluation of rhs)
        self._update(t)


    def _sample(self, t, dt):
        """Sample data from blocks that implement the 'sample' method such 
        as 'Scope', 'Delay' and the blocks that sample from a random 
        distribution at a given time 't'.
    
        Parameters
        ----------
        t : float
            time where to sample
        """
        for block in self.blocks:
            if block: block.sample(t, dt)


    def _buffer(self, t, dt):
        """Buffer states for event monitoring and internal states of blocks 
        before the timestep is taken. 

        For events, this is required to set reference for event monitoring and 
        backtracking for root finding.

        for blocks, this is required for runge-kutta integrators but also for the 
        zero crossing detection of the event handling system. The timesteps are 
        also buffered because some integrators such as GEAR-type methods need a 
        history of the timesteps.

        Parameters
        ----------
        t : float 
            evaluation time for buffering
        dt : float
            timestep
        """

        #buffer states for event detection (with timestamp)
        for event in self._get_active_events():
            event.buffer(t)

        #buffer the dummy engine (allways)
        self.engine.buffer(dt)

        #buffer internal states of stateful blocks
        for block in self._blocks_dyn:
            if block: block.buffer(dt)


    def _step(self, t, dt):
        """Performs the 'step' method for dynamical blocks with internal 
        states that have a numerical integration engine. 

        Collects the local truncation error estimates and the timestep 
        rescale factor from the error controllers of the internal 
        intergation engines if they provide an error estimate 
        (for example embedded Runge-Kutta methods).
        
        Notes
        -----
        Not to be confused with the global 'step' method, the '_step' 
        method executes the intermediate timesteps in multistage solvers 
        such as Runge-Kutta methods.
    
        Parameters
        ----------
        t : float
            evaluation time of dynamical timestepping
        dt : float
            timestep

        Returns
        -------
        success : bool 
            indicator if the timestep was successful
        max_error : float 
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        """

        #initial timestep rescale and error estimate
        success, max_error_norm, min_scale = True, 0.0, None

        #step blocks and get error estimates if available
        for block in self._blocks_dyn:

            #skip inactive blocks
            if not block: continue

            #step the block
            suc, err_norm, scl = block.step(t, dt)

            #check solver stepping success
            if not suc:
                success = False

            #update error tracking
            if err_norm > max_error_norm:
                max_error_norm = err_norm

            #track minimum relevant scale directly (avoids list allocation)
            if scl is not None:
                if min_scale is None or scl < min_scale:
                    min_scale = scl

        return success, max_error_norm, min_scale if min_scale is not None else 1.0


    # timestepping ----------------------------------------------------------------

    @deprecated(version="1.0.0", replacement="timestep")
    def timestep_fixed_explicit(self, dt=None):
        """Advances the simulation by one timestep 'dt' for explicit fixed step solvers.

        Parameters
        ----------
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """
        return self.timestep(dt, adaptive=False)


    @deprecated(version="1.0.0", replacement="timestep")
    def timestep_fixed_implicit(self, dt=None):
        """Advances the simulation by one timestep 'dt' for implicit fixed step solvers.

        Parameters
        ----------
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """
        return self.timestep(dt, adaptive=False)


    @deprecated(version="1.0.0", replacement="timestep")
    def timestep_adaptive_explicit(self, dt=None):
        """Advances the simulation by one timestep 'dt' for explicit adaptive solvers.

        Parameters
        ----------
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """
        return self.timestep(dt, adaptive=True)


    @deprecated(version="1.0.0", replacement="timestep")
    def timestep_adaptive_implicit(self, dt=None):
        """Advances the simulation by one timestep 'dt' for implicit adaptive solvers.

        Parameters
        ----------
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """
        return self.timestep(dt, adaptive=True)


    def timestep(self, dt=None, adaptive=True):
        """Advances the transient simulation by one timestep 'dt'.

        Automatic behavior selection based on selected `Solver` and `adaptive` flag:

        - Explicit solvers: Uses `_update()` for system evaluation
        - Implicit solvers: Uses `_solve()` for implicit update equation
        - Adaptive solvers (with adaptive=True): Reverts timestep if error too large
          or event not close
        - Fixed solvers (or adaptive=False): Always completes timestep, resolves
          events at detected time

        If discrete events are detected, they are handled according to stepping mode:

        - Fixed stepping: Events resolved at interpolated time within step
        - Adaptive stepping: Events approached via timestep rescaling (secant method)

        Parameters
        ----------
        dt : float
            timestep size for transient simulation
        adaptive : bool
            explicitly enable/disable adaptive timestepping; when False, adaptive
            solvers are forced to take fixed steps without error control (default True)

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """
        #solver behavior flags (adaptive only if both flag and solver support it)
        is_adaptive = adaptive and self.engine.is_adaptive
        is_implicit = not self.engine.is_explicit

        #stats tracking
        total_evals, total_solver_its = 0, 0
        error_norm, scale, success = 0.0, 1.0, True

        #default global timestep as local timestep
        if dt is None:
            dt = self.dt

        #buffer events and dynamic blocks before timestep
        self._buffer(self.time, dt)

        #solver stages iteration (skip if no dynamic blocks)
        if self._blocks_dyn:
            for time_stage in self.engine.stages(self.time, dt):

                if is_implicit:
                    #implicit: solve update equation (contains _update internally)
                    success, evals, solver_its = self._solve(time_stage, dt)
                    total_evals += evals
                    total_solver_its += solver_its

                    #adaptive implicit: revert if solver didn't converge
                    if not success and is_adaptive:
                        self._revert(self.time)
                        return False, 0.0, 0.5, total_evals + 1, total_solver_its
                else:
                    #explicit: evaluate system equation
                    self._update(time_stage)
                    total_evals += 1

                #step dynamic blocks, get error estimate
                success, error_norm, scale = self._step(time_stage, dt)

                #adaptive: revert if local truncation error too large
                if not success and is_adaptive:
                    self._revert(self.time)
                    return False, error_norm, scale, total_evals + 1, total_solver_its

        #system time after timestep
        time_dt = self.time + dt

        #evaluate system equation before event check
        self._update(time_dt)
        total_evals += 1

        #handle detected events chronologically
        for event, close, ratio in self._detected_events(time_dt):
            if is_adaptive:
                #adaptive: only resolve if close enough to event
                if close:
                    event.resolve(time_dt)
                    self._update(time_dt)
                    total_evals += 1
                else:
                    #not close: revert and use ratio as rescale
                    self._revert(self.time)
                    return False, error_norm, ratio, total_evals + 1, total_solver_its
            else:
                #fixed: resolve at interpolated time within step
                event.resolve(self.time + ratio * dt)
                self._update(time_dt)
                total_evals += 1

        #sample data after successful timestep
        self._sample(time_dt, dt)

        #increment global time
        self.time = time_dt

        return success, error_norm, scale, total_evals, total_solver_its


    def step(self, dt=None, adaptive=True):
        """Wraps 'Simulation.timestep' for backward compatibility"""
        self.logger.warning(
            "'Simulation.step' method will be deprecated with release version 1.0.0, use 'Simulation.timestep' instead!"
            )
        return self.timestep(dt, adaptive)


    # data extraction -------------------------------------------------------------

    @deprecated(version="1.0.0", reason="its against pathsims philosophy")
    def collect(self):
        """Collect all current simulation results from the internal 
        recording blocks
    
        Returns
        -------
        results : dict
        """
        scopes, spectra = {}, {}
        for block in self.blocks:
            for _category, _id, _data in block.collect():
                if _category == "scope":
                    scopes[_id] = _data
                elif _category == "spectrum":
                    spectra[_id] = _data
        return {"scopes": scopes, "spectra": spectra}


    # simulation execution --------------------------------------------------------

    def stop(self):
        """Set the flag for active simulation to 'False', intended to be
        called from the outside (for example by events) to interrupt the
        timestepping loop in 'run'.
        """
        self._active = False


    def _run_loop(self, duration, reset, adaptive, tracker=None):
        """Core simulation loop generator that yields after each timestep.

        This internal method contains the shared simulation logic used by
        'run', 'run_streaming', and 'run_realtime'. It handles initialization,
        timestepping, adaptive rescaling, and progress tracking.

        Parameters
        ----------
        duration : float
            simulation time (in time units)
        reset : bool
            reset the simulation before running
        adaptive : bool
            use adaptive timesteps if solver is adaptive
        tracker : ProgressTracker | None
            optional progress tracker for logging

        Yields
        ------
        step_info : dict
            dictionary containing 'progress', 'success', and 'dt' for each step
        """

        #set simulation active
        self._active = True

        #reset the simulation before running it
        if reset:
            self.reset()

        #make an adaptive run?
        _adaptive = adaptive and self.engine.is_adaptive

        #simulation start and end time
        start_time, end_time = self.time, self.time + duration

        #effective timestep for duration
        _dt = self.dt

        #initial system function evaluation
        self._update(self.time)

        #catch and resolve initial events
        for event, *_ in self._detected_events(self.time):

            #resolve events directly
            event.resolve(self.time)

            #evaluate system function again -> propagate event
            self._update(self.time)

        #sampling states and inputs at 'self.time == starting_time'
        self._sample(self.time, _dt)

        #main simulation loop
        while self.time < end_time and self._active:

            #advance the simulation by one (effective) timestep '_dt'
            success, error_norm, scale, *_ = self.timestep(
                dt=_dt,
                adaptive=_adaptive
                )

            #perform adaptive rescale
            if _adaptive:

                #if no error estimate and rescale -> back to default timestep
                if not error_norm and scale == 1:
                    _dt = self.dt

                #rescale due to error control
                _dt = scale * _dt

                #estimate time until next event and adjust timestep
                _dt_evt = self._estimate_events(self.time)
                if _dt_evt is not None and _dt_evt < _dt:
                    _dt = _dt_evt

                #rescale if in danger of overshooting 'end_time' at next step
                if self.time + _dt > end_time:
                    _dt = end_time - self.time

                #apply bounds to timestep after rescale
                _dt = np.clip(_dt, self.dt_min, self.dt_max)

            #compute simulation progress
            progress = np.clip((self.time - start_time) / duration, 0.0, 1.0)

            #update the tracker if provided
            if tracker:
                tracker.update(progress, success=success)

            #yield step information
            yield {'progress': progress, 'success': success, 'dt': _dt}

        #handle interrupt
        if tracker and not self._active:
            tracker.interrupt()


    def run(self, duration=10, reset=False, adaptive=True):
        """Perform multiple simulation timesteps for a given 'duration'.

        Tracks the total number of block evaluations (proxy for function
        calls, although larger, since one function call of the system equation
        consists of many block evaluations) and the total number of solver
        iterations for implicit solvers.

        Additionally the progress of the simulation is tracked by a custom
        'ProgressTracker' class that is a dynamic generator and interfaces
        the logging system.

        Parameters
        ----------
        duration : float
            simulation time (in time units)
        reset : bool
            reset the simulation before running (default False)
        adaptive : bool
            use adaptive timesteps if solver is adaptive (default True)

        Returns
        -------
        stats : dict
            stats of simulation run tracked by the 'ProgressTracker'
        """

        #initialize progress tracker
        tracker = ProgressTracker(
            total_duration=duration,
            description="TRANSIENT",
            logger=self.logger,
            log=self.log
            )

        #enter tracker context and consume the run loop
        with tracker:
            for _ in self._run_loop(duration, reset, adaptive, tracker=tracker):
                pass

        return tracker.stats


    def run_streaming(self, duration=10, reset=False, adaptive=True, tickrate=10, func_callback=None):
        """Perform simulation with streaming output at a fixed wall-clock rate.

        This method runs the simulation as fast as possible while yielding
        intermediate results at a fixed rate defined by 'tickrate'. Useful
        for real-time visualization and UI updates.

        The progress is tracked and logged using the 'ProgressTracker' class.

        Parameters
        ----------
        duration : float
            simulation time (in time units)
        reset : bool
            reset the simulation before running (default False)
        adaptive : bool
            use adaptive timesteps if solver is adaptive (default True)
        tickrate : float
            output rate in Hz, i.e., yields per second of wall-clock time
            (default 10)
        func_callback : callable | None
            callback function that is called at every tick, can be used 
            for data extraction, its return value is yielded by this generator

        Yields
        ------
        result 
            The return value of the 'func_callback' callable. 
        """

        #initialize progress tracker
        tracker = ProgressTracker(
            total_duration=duration,
            description="STREAMING",
            logger=self.logger,
            log=self.log
            )

        #streaming timing setup
        tick_interval = 1.0 / tickrate
        last_tick = time.perf_counter()

        #enter tracker context
        with tracker:

            #iterate the core simulation loop
            for step in self._run_loop(duration, reset, adaptive, tracker=tracker):

                #check if enough wall-clock time has passed
                now = time.perf_counter()
                if now - last_tick >= tick_interval:
                    last_tick = now

                    #yield intermediate results
                    yield func_callback() if callable(func_callback) else None

            #final yield with complete results
            yield func_callback() if callable(func_callback) else None


    def run_realtime(self, duration=10, reset=False, adaptive=True, tickrate=30, speed=1.0, func_callback=None):
        """Perform simulation paced to wall-clock time.

        This method runs the simulation synchronized to real time, optionally
        scaled by 'speed'. The simulation advances to match elapsed wall-clock
        time, yielding results at the rate defined by 'tickrate'.

        Useful for interactive simulations, hardware-in-the-loop testing,
        or when simulation should match real-world timing.

        The progress is tracked and logged using the 'ProgressTracker' class.

        Parameters
        ----------
        duration : float
            simulation time (in time units)
        reset : bool
            reset the simulation before running (default False)
        adaptive : bool
            use adaptive timesteps if solver is adaptive (default True)
        tickrate : float
            output rate in Hz, i.e., yields per second of wall-clock time
            (default 30)
        speed : float
            time scaling factor where 1.0 is real-time, 2.0 is twice as fast,
            0.5 is half speed (default 1.0)
        func_callback : callable | None
            callback function that is called at every tick, can be used 
            for data extraction, its return value is yielded by this generator

        Yields
        ------
        result 
            The return value of the 'func_callback' callable. 
        """

        #initialize progress tracker
        tracker = ProgressTracker(
            total_duration=duration,
            description="REALTIME",
            logger=self.logger,
            log=self.log
            )

        #realtime timing setup
        tick_interval = 1.0 / tickrate
        last_tick = time.perf_counter()
        start_wall = time.perf_counter()
        start_sim = self.time

        #enter tracker context
        with tracker:

            #create the core simulation loop generator
            loop = self._run_loop(duration, reset, adaptive, tracker=tracker)

            #realtime pacing loop
            while self._active:

                #compute target simulation time based on wall-clock
                wall_elapsed = time.perf_counter() - start_wall
                target_time = start_sim + wall_elapsed * speed

                #advance simulation until caught up with target time
                try:
                    while self.time < target_time:
                        next(loop)
                except StopIteration:
                    break

                #check if enough wall-clock time has passed for yield
                now = time.perf_counter()
                if now - last_tick >= tick_interval:
                    last_tick = now

                    #compute progress
                    progress = (self.time - start_sim) / duration

                    #yield intermediate results
                    yield func_callback() if callable(func_callback) else None

                #small sleep to avoid busy-waiting
                time.sleep(0.001)

            #final yield with complete results
            yield func_callback() if callable(func_callback) else None