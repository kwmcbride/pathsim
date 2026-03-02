#########################################################################################
##
##                   PathSim Bus Example: Closed-Loop Dual-Zone HVAC
##
##  Demonstrates bus signals flowing through a closed-loop dynamic system using
##  BusCreator, BusFunction, and BusSelector.
##
##  Two rooms (Zone A, Zone B) exchange heat with the outdoors.  A proportional
##  controller reads both zone temperatures from a structured sensor bus and
##  computes heating/cooling commands that are fed back to drive each zone's
##  thermal integrator — all wired through bus blocks.
##
##  Bus topology
##  ────────────
##
##    ┌──────────────────────────────────────────────────────────────┐
##    │                                                              │
##  [T_A, T_B]  ──►  BusCreator  ──►  sensor bus  ──►  BusFunction │
##    ▲                                                     │        │
##    │                                                     ▼        │
##    │                                               ctrl bus       │
##    │                                         {u_A, u_B}          │
##    │                                                     │        │
##    │                          BusSelector  ◄─────────────┘        │
##    │                               │                              │
##    │               u_A ◄───────────┘                              │
##    │               u_B ◄───────────┘                              │
##    │                                                              │
##    │    dT_A/dt = -(T_A - T_outdoor)/tau + u_A                   │
##    │    dT_B/dt = -(T_B - T_outdoor)/tau + u_B                   │
##    │                                                              │
##    └──────────────────────────────────────────────────────────────┘
##
##                               Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim import Bus, BusElement
from pathsim import BusCreator, BusSelector, BusFunction
from pathsim.blocks import Integrator, Scope, Function
from pathsim.solvers import SSPRK22


# PARAMETERS ============================================================================

tau       = 10.0   # thermal time constant [s] — how fast each zone reaches ambient
Kp        = 1.0    # proportional gain [°C/s per °C error]

T_outdoor = 10.0   # outdoor temperature [°C]
SP_A      = 22.0   # Zone A setpoint [°C]
SP_B      = 20.0   # Zone B setpoint [°C]

T0_A      = 15.0   # Zone A initial temperature [°C]  — below setpoint, needs heating
T0_B      = 28.0   # Zone B initial temperature [°C]  — above setpoint, needs cooling


# BUS DEFINITIONS =======================================================================

# Sensor bus: carries the current temperature of both zones
sensor_bus_def = Bus(
    name='SensorBus',
    elements=[
        BusElement('T_A', data_type='float', unit='C', description='Zone A temperature'),
        BusElement('T_B', data_type='float', unit='C', description='Zone B temperature'),
    ],
    description='Zone temperature sensor readings',
)

# Control bus: carries the heating/cooling command for each zone
ctrl_bus_def = Bus(
    name='CtrlBus',
    elements=[
        BusElement('u_A', data_type='float', unit='C/s', description='Zone A control input'),
        BusElement('u_B', data_type='float', unit='C/s', description='Zone B control input'),
    ],
    description='Zone controller outputs',
)


# PLANT BLOCKS ==========================================================================

# Each zone's thermal dynamics:  dT/dt = -(T - T_outdoor)/tau + u
def thermal_deriv(T, u):
    return -(T - T_outdoor) / tau + u

dT_A = Function(thermal_deriv)   # Zone A derivative block
dT_B = Function(thermal_deriv)   # Zone B derivative block

T_A = Integrator(T0_A)           # Zone A temperature state
T_B = Integrator(T0_B)           # Zone B temperature state


# BUS BLOCKS ============================================================================

# Pack zone temperatures into a structured sensor bus
sensor = BusCreator(sensor_bus_def)

# Proportional control law: u = Kp * (setpoint - temperature)
ctrl = BusFunction(
    func=lambda T_A, T_B: (Kp * (SP_A - T_A), Kp * (SP_B - T_B)),
    in_keys=['T_A', 'T_B'],
    out_keys=['u_A', 'u_B'],
)

# Extract scalar control commands from the control bus
ctrl_sel = BusSelector(['u_A', 'u_B'])


# RECORDING =============================================================================

scope_temps = Scope(bus=sensor_bus_def,  labels=['T_A [°C]', 'T_B [°C]'])
scope_ctrl  = Scope(bus=ctrl_bus_def,    labels=['u_A [°C/s]', 'u_B [°C/s]'])


# CONNECTIONS ===========================================================================

blocks = [dT_A, dT_B, T_A, T_B, sensor, ctrl, ctrl_sel, scope_temps, scope_ctrl]

connections = [
    # Plant dynamics: derivative function → integrator → state
    Connection(dT_A[0], T_A[0]),
    Connection(dT_B[0], T_B[0]),

    # State feeds the derivative functions (T argument)
    Connection(T_A[0], dT_A[0]),
    Connection(T_B[0], dT_B[0]),

    # Pack zone temperatures into the sensor bus
    Connection(T_A[0], sensor['T_A']),
    Connection(T_B[0], sensor['T_B']),

    # Sensor bus → BusFunction (closed-loop feedback through bus)
    Connection(sensor[0], ctrl['bus']),

    # BusFunction output → BusSelector → control scalars
    Connection(ctrl[0], ctrl_sel['bus']),

    # Control signals feed back to each zone's derivative function (u argument)
    Connection(ctrl_sel['u_A'], dT_A[1]),
    Connection(ctrl_sel['u_B'], dT_B[1]),

    # Record sensor bus and control bus directly (bus-aware Scope)
    Connection(sensor[0], scope_temps[0]),
    Connection(ctrl[0],   scope_ctrl[0]),
]


# SIMULATION ============================================================================

sim = Simulation(blocks, connections, Solver=SSPRK22, dt=0.1)


# RUN ===================================================================================

if __name__ == '__main__':

    sim.run(duration=30.0)

    t_t, y_t = scope_temps.read()
    t_c, y_c = scope_ctrl.read()

    # Analytical steady-state for P-only control:
    #   T_ss = (Kp*SP + T_outdoor/tau) / (1/tau + Kp)
    T_ss_A = (Kp * SP_A + T_outdoor / tau) / (1.0 / tau + Kp)
    T_ss_B = (Kp * SP_B + T_outdoor / tau) / (1.0 / tau + Kp)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, tight_layout=True)

    axes[0].plot(t_t, y_t[0], label='Zone A', color='tab:red')
    axes[0].plot(t_t, y_t[1], label='Zone B', color='tab:blue')
    axes[0].axhline(SP_A,   ls='--', color='tab:red',  alpha=0.4, label=f'SP_A = {SP_A}°C')
    axes[0].axhline(SP_B,   ls='--', color='tab:blue', alpha=0.4, label=f'SP_B = {SP_B}°C')
    axes[0].axhline(T_ss_A, ls=':',  color='tab:red',  alpha=0.6, label=f'T_ss_A ≈ {T_ss_A:.1f}°C')
    axes[0].axhline(T_ss_B, ls=':',  color='tab:blue', alpha=0.6, label=f'T_ss_B ≈ {T_ss_B:.1f}°C')
    axes[0].set_ylabel('Temperature [°C]')
    axes[0].set_title('Zone temperatures (P-control via bus)')
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_c, y_c[0], label='u_A (Zone A)', color='tab:red')
    axes[1].plot(t_c, y_c[1], label='u_B (Zone B)', color='tab:blue')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('Control input [°C/s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Control bus signals')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.show()
