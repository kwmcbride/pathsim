#########################################################################################
##
##                              MUTABLE PARAMETER DECORATOR
##                                 (utils/mutable.py)
##
##         Class decorator that enables runtime parameter mutation with automatic
##         reinitialization. When a decorated parameter is changed, the block's
##         __init__ is re-run with updated values while preserving engine state.
##
#########################################################################################

# IMPORTS ===============================================================================

import inspect
import functools

import numpy as np


# DECORATOR =============================================================================

def mutable(*params):
    """Class decorator that makes listed parameters trigger automatic reinitialization.

    When a parameter declared as mutable is changed at runtime, the block's ``__init__``
    is re-executed with the updated parameter values. The integration engine state is
    preserved across the reinitialization, ensuring continuity during simulation.

    A ``set(**kwargs)`` method is also generated for batched parameter updates that
    triggers only a single reinitialization.

    Parameters
    ----------
    params : str
        names of the mutable parameters (must match ``__init__`` argument names)

    Example
    -------
    .. code-block:: python

        @mutable("K", "T")
        class PT1(StateSpace):
            def __init__(self, K=1.0, T=1.0):
                self.K = K
                self.T = T
                super().__init__(
                    A=np.array([[-1.0 / T]]),
                    B=np.array([[K / T]]),
                    C=np.array([[1.0]]),
                    D=np.array([[0.0]])
                )

        pt1 = PT1(K=2.0, T=0.5)
        pt1.K = 5.0                    # auto reinitializes
        pt1.set(K=5.0, T=0.3)         # single reinitialization
    """

    def decorator(cls):

        original_init = cls.__init__

        # get all __init__ parameter names for reinit
        init_params = [
            name for name in inspect.signature(original_init).parameters
            if name != "self"
            ]

        # validate that declared mutable params exist in __init__
        for p in params:
            if p not in init_params:
                raise ValueError(
                    f"Mutable parameter '{p}' not found in "
                    f"{cls.__name__}.__init__ signature {init_params}"
                    )

        # -- install property descriptors for mutable params ---------------------------

        for name in params:
            storage = f"_p_{name}"

            def _make_property(s):
                def getter(self):
                    return getattr(self, s)

                def setter(self, value):
                    setattr(self, s, value)
                    if getattr(self, '_param_locked', False):
                        _reinit(self)

                return property(getter, setter)

            setattr(cls, name, _make_property(storage))

        # -- reinit function -----------------------------------------------------------

        def _reinit(self):
            """Re-run __init__ with current parameter values, preserving engine state."""

            # gather current values for all init params
            kwargs = {}
            for name in init_params:
                if hasattr(self, name):
                    kwargs[name] = getattr(self, name)

            # save engine state
            engine = self.engine if hasattr(self, 'engine') else None

            # re-run init (unlock to prevent recursive reinit)
            self._param_locked = False
            original_init(self, **kwargs)
            self._param_locked = True

            # restore engine
            if engine is not None:
                old_dim = len(engine)
                new_dim = len(np.atleast_1d(self.initial_value)) if hasattr(self, 'initial_value') else 0

                if old_dim == new_dim:
                    # same dimension - restore the engine directly
                    self.engine = engine
                else:
                    # dimension changed - create new engine inheriting settings
                    self.engine = type(engine).create(
                        self.initial_value,
                        parent=engine.parent,
                        from_engine=None
                        )
                    # inherit tolerances manually since from_engine=None
                    self.engine.tolerance_lte_abs = engine.tolerance_lte_abs
                    self.engine.tolerance_lte_rel = engine.tolerance_lte_rel

        # -- wrap __init__ to flip the lock after construction -------------------------

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._param_locked = True

        cls.__init__ = new_init

        # -- generate batched set() method ---------------------------------------------

        def set(self, **kwargs):
            """Set multiple parameters and reinitialize once.

            Parameters
            ----------
            kwargs : dict
                parameter names and their new values

            Example
            -------
            .. code-block:: python

                block.set(K=5.0, T=0.3)
            """
            self._param_locked = False
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._param_locked = True
            _reinit(self)

        cls.set = set

        # -- store metadata for introspection ------------------------------------------

        cls._mutable_params = params

        return cls

    return decorator
