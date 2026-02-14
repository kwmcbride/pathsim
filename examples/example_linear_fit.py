#########################################################################################
##
##            PathSim example of parameter estimation using time series data
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    Constant,
    Multiplier,
    Integrator,
    Scope,
    Adder,
    TimeSeriesSource,
)
from pathsim.solvers import SSPRK22


# DATA ==================================================================================

t_meas = np.linspace(0, 10, 21)
y_meas = np.random.uniform(0.95, 1.05, 21) * t_meas

# MODEL DEFINITION ======================================================================

#blocks that define the system
source = Source(lambda t: 1.0)  
gain = Constant(value=0.5)           
mult = Multiplier()             
integrator = Integrator()      
tsSource = TimeSeriesSource(t=t_meas, y=y_meas)
adder = Adder()
scope = Scope()

blocks = [
    source,
    gain,
    mult,
    integrator,
    adder,
    scope,
    tsSource,

]

#the connections between the blocks
connections = [
    Connection(source[0], mult[0]),
    Connection(gain[0], mult[1]),
    Connection(mult[0], integrator[0]),
    Connection(integrator[0], adder[0]),
    Connection(adder[0], scope[0]),
    Connection(tsSource[0], scope[1]),
]

# initialize simulation with the blocks, connections, timestep and logging enabled
sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=0.0001,
    tolerance_lte_abs=1e-08,
    tolerance_fpi=1e-10,
    log=False,
)

# Run Example ===========================================================================

if __name__ == '__main__':

    # Parameter estimation imports
    from pathsim.opt import ParameterEstimator, TimeSeriesData

    # run the simulation for some time
    sim.run(duration=10.0)

    # sim.plot()
    
    # plt.show()
    
    # create parameter estimator instance
    est = ParameterEstimator(
        simulator=sim,
        adaptive=True,
    )

    # Add block parameters to estimate - here the gain and the initial value of the integrator are estimated
    est.add_block_parameter(gain, 'value', id='gain', value=3)
    est.add_block_parameter(integrator, 'initial_value', id='integrator', bounds=(0.0, 5), value=2)

    print(est.parameters)

    # create TimeSeriesData explicitly (can use TimeSeriesSource directly as well)
    meas = TimeSeriesData(time=t_meas, data=y_meas, name="y_meas")
    # meas = tsSource._series

    # register measurement + model output mapping
    est.add_timeseries(meas, signal=scope[0], sigma=1.0)

    # run the fitting routine
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    # Plot
    t_pred, y_pred = est.simulate(fit.x)

    plt.figure(figsize=(8, 5))
    plt.plot(t_meas, y_meas, 'o', ms=5, alpha=0.6, label='Measured')
    plt.plot(t_pred, y_pred, '-', lw=2, label=f'Fit') #: K={fit.x[0]:.3f}, x0={fit.x[1]:.3f}')
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.title('Parameter Estimation with Parameter Objects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    est.display()