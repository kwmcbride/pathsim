#########################################################################################
##
##  PathSim example: multi-experiment nested Schur optimisation
##
##  Model:   First-order decay with per-experiment amplitude
##
##      y'(t) = -k * y(t)        (same decay rate k across all experiments)
##      y(0)  = 1                (unit initial condition — amplitude handled separately)
##      output = amp_i * y(t)    (per-experiment scale)
##
##  Analytical solution:  output_i(t) = amp_i * exp(-k * t)
##
##  Parameters
##  ──────────
##  Global (shared across all experiments):
##      k      [1/s]  decay rate   — Constant block value, via add_global_block_parameter
##
##  Local (one per experiment):
##      amp_i  [ ]    output amplitude — Amplifier gain, via add_local_block_parameter
##
##  Nested Schur algorithm  (ParameterEstimator.fit_nested)
##  ────────────────────────────────────────────────────────
##  Outer loop:  Gauss-Newton on k using the reduced Jacobian
##
##      J_red_i  = P_{L,i} J_{G,i}     (project out local directions)
##      r_red_i  = P_{L,i} r_i*        (project residuals)
##      Δk = -(J_red^T J_red)^{-1} J_red^T r_red   (Schur complement step)
##
##  Inner loop (per outer iteration):
##      argmin_{amp_i}  ||r_i(k, amp_i)||²   — independent per experiment,
##                                              warm-started from previous iterate
##
##  After fitting, sensitivity() computes:
##    · Full joint FIM over all parameters
##    · SchurResult: effective FIM for k after marginalising the amp_i
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Scope
from pathsim.blocks.sources import Constant
from pathsim.blocks.ode import ODE
from pathsim.solvers import SSPRK22

from pathsim.opt import ParameterEstimator, TimeSeriesData


# TRUE PARAMETER VALUES =================================================================

TRUE_K    = 0.4          # [1/s]  decay rate  — shared across experiments
TRUE_AMPS = [8.0, 3.5, 1.2]   # output amplitudes  — one per experiment


# SIMULATION MODEL ======================================================================
#
#   Constant(k) ──→ ODE(y'=-k*y, y0=1) ──→ Amplifier(gain=amp_i) ──→ Scope
#
# The decay rate k enters the ODE as an input from the Constant block so that
# add_global_block_parameter can target it across all deep-copied experiments.
# Each experiment's Amplifier gain is the local amplitude parameter.

k_const = Constant(value=TRUE_K)
ode     = ODE(
    func=lambda x, u, t: np.array([-u[0] * x[0]]),
    initial_value=[1.0],
)
amp_out = Amplifier(gain=1.0)
scope   = Scope(labels=["y(t)"])

sim = Simulation(
    [k_const, ode, amp_out, scope],
    [
        Connection(k_const[0], ode[0]),
        Connection(ode[0],     amp_out[0]),
        Connection(amp_out[0], scope[0]),
    ],
    Solver=SSPRK22,
    dt=0.05, dt_min=1e-12,
    tolerance_lte_rel=1e-5,
    tolerance_lte_abs=1e-8,
    log=False,
)


# SYNTHETIC MEASUREMENTS ================================================================

def analytical(t, k, amp):
    return amp * np.exp(-k * t)


t_meas = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
rng    = np.random.default_rng(17)

measurements = []
for amp in TRUE_AMPS:
    y_clean = analytical(t_meas, TRUE_K, amp)
    y_noisy = y_clean * (1.0 + 0.05 * rng.standard_normal(len(t_meas)))
    measurements.append(TimeSeriesData(time=t_meas, data=np.maximum(y_noisy, 0.0),
                                       name=f"amp={amp}"))


# RUN EXAMPLE ===========================================================================

if __name__ == '__main__':

    n_exp = len(TRUE_AMPS)

    # ── Build estimator ──────────────────────────────────────────────────────
    est = ParameterEstimator(simulator=sim, adaptive=True)
    for _ in range(1, n_exp):
        est.add_experiment(sim, copy_sim=True, adaptive=True)

    for i, meas in enumerate(measurements):
        est.add_timeseries(meas, signal=scope[0], experiment=i)

    # Global: Constant.value = k  (SharedBlockParameter, writes to all copies)
    est.add_global_block_parameter(
        "Constant", "value",
        value=0.2, bounds=(0.01, 5.0),
        param_id="k",
    )

    # Local: Amplifier.gain = amp_i  (one per experiment)
    for i in range(n_exp):
        est.add_local_block_parameter(
            i, "Amplifier", "gain",
            value=2.0, bounds=(0.05, 30.0),
            param_id=f"amp{i}",
        )

    # ── Nested Schur fit ─────────────────────────────────────────────────────
    #
    # fit_nested() runs:
    #   Outer: Gauss-Newton on k via the reduced Jacobian J_red = P_{L,i} J_{G,i}
    #   Inner: independent least_squares on each amp_i for fixed k
    #
    print("Running nested Schur fit ...")
    
    fit = est.fit_nested(
        max_outer_nfev=30,
        max_inner_nfev=20,
        loss="soft_l1",
        f_scale=0.3,
        verbose=1,
    )
    print()
    est.display()
    print(f"\n  True values:  k={TRUE_K}  amps={TRUE_AMPS}")

    # ── Sensitivity + Schur ──────────────────────────────────────────────────
    #
    # sensitivity() uses the full Jacobian to compute:
    #   · Joint FIM over all parameters (k, amp_0, amp_1, amp_2)
    #   · SchurResult: effective FIM for k after marginalising the amp_i
    #
    print("\nComputing post-fit sensitivity + Schur complement ...")
    sens = est.sensitivity()
    sens.display()

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig_fit, _ = est.plot_fit(
        fit.x,
        title="Decay fit (nested Schur)",
        xlabel="Time [s]",
        ylabel="y(t)",
    )

    fig_sens, _ = sens.plot()

    if sens.schur is not None:
        fig_schur, _ = sens.schur.plot()

    plt.show()
