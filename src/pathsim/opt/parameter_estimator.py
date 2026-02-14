"""
#########################################################################################
##
##                 PARAMETER ESTIMATION TOOLKIT (PathSim Extension Layer)
##                           (parameter_estimator.py)
##
##                              Kevin McBride 2026
##
#########################################################################################

OVERVIEW
--------
This module provides a lightweight parameter-estimation layer for PathSim.

It is intended for iterative optimization loops where a PathSim `Simulation`
is evaluated repeatedly with different parameter values.

FEATURES
--------
- Parameter declaration with optional bounds and transforms
- Measurement handling via time-aligned `TimeSeriesData`
- Model output extraction from PathSim `Scope` via `ScopeSignal`
- Stateful simulation reset + post-reset hooks via `SimRunner`
- SciPy-based fitting (`least_squares` and `minimize`)

NOTES
-----
- PathSim simulations are stateful; objective evaluation must reset and rerun.
- Outputs must be read from a `Scope` (or compatible `.read()` provider).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize as sci_opt

from ..blocks._block import Block
from .timeseries_data import TimeSeriesData

# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER DECLARATION
# ═════════════════════════════════════════════════════════════════════════════

class Parameter:
    
    """Unified estimation parameter.

    A parameter can either be:
    - a free scalar used in user-defined Python code (e.g., ODE closures)
    - a block-bound parameter mapped to a PathSim block attribute

    Optional transforms allow fitting in an unconstrained space while applying a
    physically valid value to the model (e.g., `np.exp` to enforce positivity).

    Parameters
    ----------
    name : str
        Parameter identifier.
    value : float
        Initial value in optimizer space.
    bounds : tuple[float, float]
        Lower/upper bounds in optimizer space.
    transform : callable, optional
        Value transform applied when the parameter is read/applied.
    block : object, optional
        Target block/object for block-bound parameters.
    attribute : str, optional
        Target attribute path (supports dotted access).

    Notes
    -----
    For block-bound parameters, calling `set(...)` applies the transformed value
    to the target object immediately.
    """

    def __init__(
        self,
        name: str,
        value: float = 1.0,
        bounds: Tuple[float, float] = (-np.inf, np.inf),
        transform: Optional[Callable[[float], float]] = None,
        block: Optional[Any] = None,
        attribute: Optional[str] = None,
    ):
        self.name = name
        self._value = float(value)
        self.bounds = bounds
        self.transform = transform
        
        # Block parameter attributes
        self.block = block
        self.attribute = attribute
        
        # Validate configuration
        if block is not None and attribute is None:
            raise ValueError("attribute must be provided when block is specified")
        
        # Determine type
        self._is_block_param = block is not None
        
        # Initialize - apply to block if needed
        self.set(value)
    
    
    @property
    def is_block_parameter(self) -> bool:
        """Check if this is a block parameter."""
        return self._is_block_param
    
    
    @property
    def is_free_parameter(self) -> bool:
        """Check if this is a free parameter."""
        return not self._is_block_param
    
    
    @property
    def value(self) -> float:
        """Current parameter value (in optimizer space)."""
        return self._value
    
    
    @value.setter
    def value(self, new_value: float) -> None:
        self.set(new_value)
    
    
    def __call__(self) -> float:
        """Return the parameter value in model space (after optional transform)."""
        return self.transform(self._value) if self.transform is not None else self._value
    
    
    def set(self, value: float) -> None:
        """Set optimizer-space value and push to target (if block-bound)."""
        self._value = float(value)
        transformed = self()
        
        if self._is_block_param:
            # Apply to block attribute
            obj = self.block
            attrs = self.attribute.split('.')
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], transformed)
    
    
    def __repr__(self) -> str:
        if self._is_block_param:
            return (f"Parameter(name={self.name!r}, value={self._value}, "
                    f"block={type(self.block).__name__}, "
                    f"attribute={self.attribute!r}, bounds={self.bounds})")
        else:
            return (f"Parameter(name={self.name!r}, value={self._value}, "
                    f"bounds={self.bounds})")


#TODO: remove these in future version?
def BlockParameter(block, attribute, name=None, **kwargs):
    """Factory for block-bound Parameters.

    Parameters
    ----------
    block : object
        Target block/object.
    attribute : str
        Attribute path (e.g. "value" or "config.gain").

    Returns
    -------
    Parameter
        A block-bound parameter.
    """
    if name is None:
        name = f'{type(block).__name__}.{attribute}'
    return Parameter(name=name, block=block, attribute=attribute, **kwargs)


def FreeParameter(name, **kwargs):
    """Factory for free (non-block) Parameters."""
    return Parameter(name=name, **kwargs)



@dataclass
class ScopeSignal:
    
    """Scope output selection.

    Reads a single port from an object that provides `read() -> (t, y)`.

    Parameters
    ----------
    scope : object
        Scope-like provider with `.read()`.
    port : int
        Port index to extract from multi-port scopes.
    """

    scope: object
    port: int = 0


    def _extract(self, t, y):
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 1:
            # Single signal already
            y_port = y_arr
        elif y_arr.ndim == 2:
            # Support (n_ports, n_samples) and (n_samples, n_ports)
            if y_arr.shape[0] == t_arr.size and y_arr.shape[1] != t_arr.size:
                y_arr = y_arr.T
            y_port = y_arr[self.port, :]
        else:
            raise ValueError("ScopeSignal supports 1D or 2D scope data only")

        return t_arr, np.asarray(y_port, dtype=float).reshape(-1)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read `(t, y)` from the scope and extract the selected port."""
        t, y = self.scope.read()
        return self._extract(t, y)


@dataclass
class SimRunner:
    
    """Simulation runner adapter.

    Provides a callable/reset+run interface suitable for optimizers.

    Parameters
    ----------
    sim : object
        PathSim simulation instance.
    output : ScopeSignal
        Output extractor (typically used outside `run()`).
    duration : float
        Simulation duration per evaluation.
    post_reset_hooks : list[callable], optional
        Hooks executed immediately after `sim.reset(...)`.
    pre_run : callable, optional
        Hook executed before `sim.run(...)`.
    adaptive : bool
        Enable adaptive stepping (if supported by PathSim simulation).
    reset_time : float
        Reset time passed into `sim.reset(time=...)`.
    suppress_reset_log : bool
        Temporarily disable sim logging during reset/run for cleaner optimization output.
    """

    sim: object
    output: ScopeSignal
    duration: float
    post_reset_hooks: List[Callable[[], None]] | None = None
    pre_run: Callable[[], None] | None = None
    adaptive: bool = False
    reset_time: float = 0.0
    suppress_reset_log: bool = True


    def after_reset(self, fn: Callable[[], None]) -> "SimRunner":
        """Register a hook executed after every `sim.reset(...)`."""
        if self.post_reset_hooks is None:
            self.post_reset_hooks = []
        self.post_reset_hooks.append(fn)
        return self


    def before_run(self, fn: Callable[[], None]) -> "SimRunner":
        """Set a hook executed immediately before `sim.run(...)`."""
        self.pre_run = fn
        return self


    def __call__(self, _x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the simulation and return the configured output."""
        self.run()
        return self.output.read()


    def run(self) -> None:
        """Reset and run the simulation (no output read)."""
        old_log = None
        if self.suppress_reset_log and hasattr(self.sim, 'log'):
            old_log = self.sim.log
            self.sim.log = False

        try:        
            self.sim.reset(time=self.reset_time)

            # Run after_reset hooks
            if self.post_reset_hooks is not None:
                for hook in self.post_reset_hooks:
                    hook()

            if self.pre_run is not None:
                self.pre_run()

            self.sim.run(duration=self.duration, reset=False, adaptive=self.adaptive)
        finally:
            # Restore original logging state
            if old_log is not None:
                self.sim.log = old_log


@dataclass
class EstimatorResult:
    
    """Optimization result container."""
    x: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str


class ParameterEstimator:
    
    """Parameter estimation driver.

    Parameters
    ----------
    parameters : list[Parameter], optional
        Parameters to estimate.
    simulator : object
        PathSim simulation or a runner providing `run()` / callable interface.
    measurement : TimeSeriesData | list[TimeSeriesData], optional
        Measurement(s) to fit.
    outputs : ScopeSignal | list[ScopeSignal], optional
        Output mapping(s) corresponding to measurements.
    duration : float, optional
        Override simulation duration; otherwise derived from measurements.
    adaptive : bool
        Enable adaptive stepping (if supported).
    pre_run : callable, optional
        Hook executed before each sim run.
    sigma : float | np.ndarray | list, optional
        Measurement noise scaling for residual normalization.
    """

    def __init__(
        self,
        parameters: Optional[List] = None,
        simulator=None,
        measurement=None,
        outputs=None,
        duration: Optional[float] = None,
        adaptive: bool = False,
        pre_run: Optional[Callable] = None,
        # after_reset: Optional[Callable] = None,
        sigma: Union[float, np.ndarray] = None,
    ):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        
        # Handle single or multiple measurements/outputs
        if measurement is None:
            measurement = []
        
        if isinstance(measurement, TimeSeriesData):
            self.measurements = [measurement]
            self.outputs = [outputs] if outputs is not None else [None]
        else:
            self.measurements = measurement
            self.outputs = outputs if outputs is not None else [None] * len(measurement)
        
        # Create runner for simulation
        if hasattr(simulator, 'run'):
            self.runners = [SimRunner(
                sim=simulator,
                output=None,  # outputs are read from ScopeSignal
                duration=duration,
                adaptive=adaptive,
                pre_run=pre_run,
            )]
        else:
            self.runners = [simulator]
            
        self._update_duration_from_measurements()
        
        if sigma is None:
            sigma = []
        self.sigma = sigma
        

    def add_parameters(self, params) -> "ParameterEstimator":
        self.parameters.extend(params)
        return self


    def add_block_parameter(self, block, param_name, value=None, bounds=(-np.inf, np.inf), id=None) -> "ParameterEstimator":
        param = block_param_to_var(block, param_name, value=value, bounds=bounds, id=id)
        self.parameters.append(param)
        return self

    
    def add_measurement(self, t, y, *, name: str = "measurement", sigma: Optional[float] = None) -> "ParameterEstimator":
        self.measurements.append(TimeSeriesData(time=t, data=y, name=name))
        self.sigma.append(float(sigma) if sigma is not None else None)
        self._update_duration_from_measurements()
        return self
    
    
    def add_output(self, scope, port: int = 0) -> "ParameterEstimator":
        self.outputs.append(ScopeSignal(scope=scope, port=int(port)))
        self._normalize_outputs()
        return self
        
        
    @staticmethod
    def _scope_port_from_signal(signal: object) -> tuple[object, int]:
        """Extract `(scope, port)` mapping.

        Supports:
        - explicit `(scope, port)` tuples
        - PathSim PortReference-like objects with `.block` and `.ports`
        """
        # Explicit (scope, port)
        if isinstance(signal, (tuple, list)) and len(signal) == 2:
            return signal[0], int(signal[1])

        # PathSim PortReference: has `.block` and `.ports` (ports is list-like of ints)
        blk = getattr(signal, "block", None)
        ports = getattr(signal, "ports", None)
        if blk is not None and ports is not None:
            if len(ports) != 1:
                raise ValueError(f"Expected PortReference with a single port, got {len(ports)}.")
            return blk, int(ports[0])

        raise TypeError(
            f"Unsupported signal type {type(signal).__name__}. "
            "Use signal=scope[0] (PortReference) or signal=(scope, port)."
        )


    def add_timeseries(
        self,
        ts: "TimeSeriesData",
        *,
        scope: object | None = None,
        port: int | None = None,
        signal: object | None = None,
        sigma: float | None = None,
    ) -> "ParameterEstimator":
        """Add a measurement and map it to a scope port.

        Parameters
        ----------
        ts : TimeSeriesData
            Measurement series.
        scope : object, optional
            Scope-like object with `.read() -> (t, y)`.
        port : int, optional
            Scope port index.
        signal : object, optional
            Either `scope[port]` (PortReference) or `(scope, port)`.
        sigma : float, optional
            Residual scaling for this series.

        Returns
        -------
        ParameterEstimator
            Self (for chaining).
        """
        if not isinstance(ts, TimeSeriesData):
            raise TypeError(f"add_timeseries expects TimeSeriesData, got {type(ts).__name__}")

        if signal is not None:
            if scope is not None or port is not None:
                raise ValueError("Pass either `signal=` OR (`scope=`, `port=`), not both.")
            scope, port = self._scope_port_from_signal(signal)

        if scope is None or port is None:
            raise ValueError("You must pass either scope=..., port=... or signal=scope[0].")

        if not (hasattr(scope, "read") and callable(getattr(scope, "read"))):
            raise TypeError(f"`scope` must provide .read() -> (t, y); got {type(scope).__name__}")

        self.measurements.append(ts)
        self.sigma.append(float(sigma) if sigma is not None else None)
        self._update_duration_from_measurements()

        if self.outputs is None:
            self.outputs = []
        self.outputs.append(ScopeSignal(scope=scope, port=int(port)))
        self._normalize_outputs()
        return self
    
        
    def _normalize_outputs(self) -> None:
        self._outputs = []
        for out in self.outputs:
            if out is None:
                self._outputs.append(None)
            elif isinstance(out, ScopeSignal):
                self._outputs.append(out)
            elif isinstance(out, tuple) and len(out) == 2:
                scope, port = out
                self._outputs.append(ScopeSignal(scope=scope, port=int(port)))
            else:
                self._outputs.append(ScopeSignal(scope=out, port=0))
                
                
    def _update_duration_from_measurements(self) -> None:
        if self.measurements:
            self.duration = max(meas.time.max() for meas in self.measurements)
            if self.runners and hasattr(self.runners[0], "duration"):
                self.runners[0].duration = self.duration


    def apply(self, x: np.ndarray) -> None:
        """Apply optimizer parameter vector to the model."""
        # self._last_x = np.asarray(x, dtype=float).copy()
        
        for param, val in zip(self.parameters, x):
            val_float = float(val)

            # 1. Update the parameter object using its public API
            # if hasattr(param, 'set'):
            param.set(val_float)
           
            if hasattr(param, '_value'):
                object.__setattr__(param, '_value', val_float)

        # Update source-only blocks (no inputs) after parameter changes.
        if hasattr(self, 'runners') and self.runners:
            sim = self.runners[0].sim
            for block in sim.blocks:
                if hasattr(block, 'num_inputs') and block.num_inputs == 0:
                    block.update(0.0)

    
    def simulate(self, x: np.ndarray, output_idx: int = 0):
        self._update_duration_from_measurements()
        self.apply(x)

        if hasattr(self.runners[0], "run") and callable(getattr(self.runners[0], "run")):
            self.runners[0].run()
            out = self._outputs[output_idx]
            if out is None:
                raise ValueError("Output cannot be None for simulation.")
            return out.read()

        self.runners[0].reset(time=0.0)

        return self.runners[0](None)
    
    
    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute stacked residual vector across all mapped outputs."""
        self._update_duration_from_measurements()
        self.apply(x)

        if not hasattr(self.runners[0], "run"):
            t_sim, y_sim = self.runners[0](None)
            t_sim = np.asarray(t_sim, dtype=float).reshape(-1)
            y_sim = np.asarray(y_sim, dtype=float).reshape(-1)
            y_pred = np.interp(self.measurements[0].time, t_sim, y_sim)
            sigma = self.sigma if not isinstance(self.sigma, (list, np.ndarray)) else self.sigma[0]
            return (y_pred - self.measurements[0].data) / sigma

        # Run sim once
        self.runners[0].run()

        all_residuals = []
        for idx, meas in enumerate(self.measurements):
            out = self._outputs[idx]
            if out is None:
                raise ValueError("Output cannot be None for multi-output estimation.")

            t_out, y_out = out.read()
            t_out = np.asarray(t_out, dtype=float).reshape(-1)
            y_out = np.asarray(y_out, dtype=float).reshape(-1)

            y_pred = np.interp(meas.time, t_out, y_out)
            sigma_i = self.sigma[idx] if isinstance(self.sigma, (list, np.ndarray)) else self.sigma
            all_residuals.append((y_pred - meas.data) / sigma_i)

        return np.concatenate(all_residuals)
    
    
    def display(self):  
        
        for param in self.parameters:
            print(f"{param.name}: {param()}")
    
    
    def fit(
        self,
        *,
        x0: Optional[Sequence[float]] = None,
        bounds: Tuple[Sequence[float], Sequence[float]] | None = None,
        loss: str = "linear",
        f_scale: float = 1.0,
        max_nfev: int = 80,
        verbose: int = 0,
        method: str = "least_squares",
        constraints: Optional[List[dict]] = None,
    ) -> EstimatorResult:
        """Fit parameters using SciPy optimizers.

        Parameters
        ----------
        x0 : sequence of float, optional
            Initial optimizer-space parameter vector.
        bounds : (lower, upper), optional
            Bounds in optimizer space.
        loss : str
            Loss for `scipy.optimize.least_squares`.
        f_scale : float
            Loss scale for robust losses.
        max_nfev : int
            Max function evaluations / iterations (depending on solver).
        verbose : int
            Verbosity level.
        method : str
            Solver selection: "least_squares" or a `scipy.optimize.minimize` method.
        constraints : list of dict, optional
            Constraint definitions for `scipy.optimize.minimize`.

        Notes
        -----
        - General constraints are only supported via `minimize` methods such as
          'SLSQP', 'trust-constr', or 'COBYLA'.
        """
        # Auto-extract x0 and bounds from Parameters
        if x0 is None:
            x0 = [p.value for p in self.parameters]
        x0_arr = np.asarray(x0, dtype=float)
        
        if bounds is None:
            lower = np.array([p.bounds[0] for p in self.parameters], dtype=float)
            upper = np.array([p.bounds[1] for p in self.parameters], dtype=float)
            bounds = (lower, upper)
        
        bounds_list = list(zip(bounds[0], bounds[1]))
        
        
        # Objective function (scalar cost from residuals)
        def objective(x):
            r = self.residuals(x)
            return 0.5 * np.sum(r**2)
        
        
        def _callback(xk):
            if verbose > 0:
                print(f"iter x={xk}, obj={objective(xk)}")
        
        
        # Choose solver
        if method == "least_squares":
            if constraints is not None:
                raise ValueError(
                    "least_squares does not support general constraints. "
                    "Use method='SLSQP' or 'trust-constr' instead."
                )
            
            res = sci_opt.least_squares(
                self.residuals,
                x0=x0_arr,
                bounds=bounds,
                loss=loss,
                f_scale=float(f_scale),
                max_nfev=int(max_nfev),
                verbose=int(verbose),
            )
            
            # self.apply(res.x)
            
            return EstimatorResult(
                x=res.x,
                cost=float(res.cost),
                nfev=int(res.nfev),
                success=bool(res.success),
                message=str(res.message),
            )
        
        elif method in ["SLSQP", "trust-constr", "COBYLA"]:
            # Methods that support constraints
            res = sci_opt.minimize(
                objective,
                x0=x0_arr,
                bounds=bounds_list,
                method=method,
                constraints=constraints,  # ← Pass constraints here
                callback=_callback,
                options={'maxiter': max_nfev, 'disp': verbose > 0}
            )
           
            # self.apply(res.x)
            # Compute final cost
            final_residuals = self.residuals(res.x)
            cost = 0.5 * np.sum(final_residuals**2)
            
            return EstimatorResult(
                x=res.x,
                cost=float(cost),
                nfev=int(res.nfev) if hasattr(res, 'nfev') else max_nfev,
                success=bool(res.success),
                message=str(res.message),
            )
        
        else:
            # Other methods (L-BFGS-B, differential_evolution, etc.)
            if constraints is not None:
                raise ValueError(
                    f"Method '{method}' does not support general constraints. "
                    "Use 'SLSQP', 'trust-constr', or 'COBYLA'."
                )
            
            res = sci_opt.minimize(
                objective,
                x0=x0_arr,
                bounds=bounds_list,
                method=method,
                callback=_callback,
                options={'maxiter': max_nfev, 'disp': True}
            )
            
            # self.apply(res.x)
            
            final_residuals = self.residuals(res.x)
            cost = 0.5 * np.sum(final_residuals**2)
            
            return EstimatorResult(
                x=res.x,
                cost=float(cost),
                nfev=int(res.nfev) if hasattr(res, 'nfev') else max_nfev,
                success=bool(res.success),
                message=str(res.message),
            )
            
            
def free_param_to_var(param_name, value=None, bounds=(-np.inf, np.inf)):
    """Create a free (non-block) parameter for estimation.

    Parameters
    ----------
    param_name : str
        Parameter name.
    value : float
        Initial value (optimizer space).
    bounds : tuple[float, float]
        Lower/upper bounds.

    Returns
    -------
    Parameter
        Free parameter instance.
    """
    if value is None:   
        raise ValueError("Initial value must be provided for free parameters.")

    return FreeParameter(
        name=param_name,
        value=value,
        bounds=bounds,
    )   


def block_param_to_var(block, param_name, value=None, bounds=(-np.inf, np.inf), id=None):
    """Create a block-bound parameter for estimation.

    Parameters
    ----------
    block : pathsim.blocks.Block
        Target block.
    param_name : str
        Attribute name on the block.
    value : float, optional
        Initial value; defaults to current block attribute.
    bounds : tuple[float, float]
        Lower/upper bounds.
    id : str, optional
        Identifier prefix for the parameter name.

    Returns
    -------
    Parameter
        Block-bound parameter instance.
    """

    if not hasattr(block, param_name):
        raise AttributeError(f"Block '{block}' has no attribute '{param_name}'")
    
    if id is None:
        id = f"{block.__class__.__name__}.{param_name}"
    else:
        id = f"{id}.{param_name}"

    if value is None:
        value = getattr(block, param_name)

    return BlockParameter(
        block=block,
        attribute=param_name,
        name=id,
        value=value,
        bounds=bounds,
    )
    
    
def get_block_dict(sim, globals_dict=None):
    """Build lookup dictionaries for blocks contained in a simulation.

    Parameters
    ----------
    sim : pathsim.Simulation
        Simulation containing blocks.
    globals_dict : dict, optional
        Namespace mapping to search. If omitted, uses this module's globals.

    Returns
    -------
    (dict, dict)
        `(block_dict, lookup_dict)` where `block_dict[name] -> block` and
        `lookup_dict[block] -> name`.

    Notes
    -----
    This helper depends on the namespace provided; if called from imported code,
    pass the caller's `globals()` explicitly.
    """
    block_dict = {}
    lookup_dict = {}
    
    # This fails to find blocks if this function is called from an imported module, since it looks at the global scope of that module instead of the caller's global scope where the blocks are defined. I don't have a better way to get the caller's globals() dict without moding pathsim to pass it in.
    if globals_dict is None:
        globals_dict = globals()
    
    for name, obj in globals_dict.items():
        if isinstance(obj, Block) and obj in sim.blocks:
            block_dict[name] = obj
            lookup_dict[obj] = name
            
    return block_dict, lookup_dict