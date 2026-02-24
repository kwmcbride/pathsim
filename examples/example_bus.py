#########################################################################################
##
##                          PathSim Bus Example: Ideal Gas Law
##
##  Demonstrates the Bus, BusCreator, and BusSelector blocks.
##
##  Models two gas chambers whose pressure is computed from the ideal gas law
##
##      P = n * R * T / V
##
##  using separate temperature and volume integrators.  All sensor readings
##  are bundled into a structured bus and then selectively extracted for
##  recording.
##
##                               Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import (
    Integrator, Adder, Amplifier, Scope, Constant, Function,
    BusCreator, BusSelector,
)
from pathsim.bus import Bus, BusElement
from pathsim.solvers import SSPRK22


# BUS DEFINITION ========================================================================

# Each chamber reports temperature [K], volume [L], and pressure [kPa].
# Define the bus structure once and reuse it for both chambers.

temp_elem     = BusElement('Temperature', data_type='float', unit='K',   description='Gas temperature')
volume_elem   = BusElement('Volume',      data_type='float', unit='L',   description='Chamber volume')
pressure_elem = BusElement('Pressure',    data_type='float', unit='kPa', description='Gas pressure P=nRT/V')

chamber_bus_def = Bus(
    name='ChamberBus',
    elements=[temp_elem, volume_elem, pressure_elem],
    description='State of one ideal-gas chamber',
)


# PHYSICAL CONSTANTS ====================================================================

R = 8.314      # J / (mol·K)  — universal gas constant
n = 1.0        # mol           — amount of gas (same for both chambers)


# CHAMBER SUBSYSTEM =====================================================================
#
# Encapsulates the dynamics of a single ideal-gas chamber as a Subsystem.
# Inputs (via Interface ports):
#   port 0 — heat flow  dT/dt  [K/s]
#   port 1 — piston speed dV/dt [L/s]
# Output bus (port 0):
#   flat float64 array with layout defined by chamber_bus_def

def make_chamber(T0, V0):
    """Return a Subsystem modelling one ideal-gas chamber.

    Parameters
    ----------
    T0 : float  Initial temperature [K]
    V0 : float  Initial volume [L]
    """

    # Interface block — two scalar inputs, one bus output
    iface = Interface()

    # Integrators for temperature and volume
    T_int = Integrator(T0)   # port 0 of Interface → dT/dt
    V_int = Integrator(V0)   # port 1 of Interface → dV/dt

    # Pressure computed from T and V:  P = n*R*T / V
    # Function block unpacks inputs as positional args
    pressure_fn = Function(lambda T, V: n * R * T / V)

    # Pack the three signals into the chamber bus
    creator = BusCreator(chamber_bus_def)

    sub_blocks = [iface, T_int, V_int, pressure_fn, creator]

    sub_connections = [
        # Drive integrators from the Interface inputs
        Connection(iface[0], T_int),
        Connection(iface[1], V_int),
        # Pressure block reads T (port 0) and V (port 1)
        Connection(T_int, pressure_fn[0]),
        Connection(V_int, pressure_fn[1]),
        # Pack T, V, P into the bus
        Connection(T_int,       creator['Temperature']),
        Connection(V_int,       creator['Volume']),
        Connection(pressure_fn, creator['Pressure']),
        # Expose bus on Interface output port 0
        Connection(creator[0], iface[0]),
    ]

    return Subsystem(sub_blocks, sub_connections)


# MODEL SETUP ===========================================================================

# Chamber A: warm, small  →  high pressure
# Chamber B: cool, large  →  low pressure
chamber_A = make_chamber(T0=400.0, V0=5.0)
chamber_B = make_chamber(T0=300.0, V0=20.0)

# Constant heat/volume-change rates driving each chamber
dT_A = Constant( 5.0)   # K/s  — heating
dV_A = Constant( 0.0)   # L/s  — slight expansion


dT_B = Constant(-2.0)   # K/s  — cooling
dV_B = Constant( 0.5)   # L/s  — faster expansion

# Extract selected signals from each chamber's bus
sel_A = BusSelector(keys=['Temperature', 'Pressure'])
sel_B = BusSelector(keys=['Temperature', 'Pressure'])

# Record extracted signals
scope_A = Scope(labels=['T_A [K]', 'P_A [kPa]'])
scope_B = Scope(labels=['T_B [K]', 'P_B [kPa]'])


# CONNECTIONS ===========================================================================

blocks = [
    dT_A, dV_A, chamber_A, sel_A, scope_A,
    dT_B, dV_B, chamber_B, sel_B, scope_B,
]

connections = [
    # Drive chamber A
    Connection(dT_A[0], chamber_A[0]),
    Connection(dV_A[0], chamber_A[1]),
    # Drive chamber B
    Connection(dT_B[0], chamber_B[0]),
    Connection(dV_B[0], chamber_B[1]),
    # Route chamber bus → selector → scope
    Connection(chamber_A[0], sel_A[0]),
    Connection(sel_A['Temperature'], scope_A[0]),
    Connection(sel_A['Pressure'],    scope_A[1]),
    Connection(chamber_B[0], sel_B[0]),
    Connection(sel_B['Temperature'], scope_B[0]),
    Connection(sel_B['Pressure'],    scope_B[1]),
]


# SIMULATION ============================================================================

sim = Simulation(blocks, connections, Solver=SSPRK22, dt=0.1)


# RUN ===================================================================================

if __name__ == '__main__':

    sim.run(duration=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)

    t_A, y_A = scope_A.read()
    t_B, y_B = scope_B.read()

    axes[0, 0].plot(t_A, y_A[0])
    axes[0, 0].set_title('Chamber A — Temperature')
    axes[0, 0].set_ylabel('T [K]')
    axes[0, 0].grid()

    axes[1, 0].plot(t_A, y_A[1])
    axes[1, 0].set_title('Chamber A — Pressure')
    axes[1, 0].set_ylabel('P [kPa]')
    axes[1, 0].set_xlabel('time [s]')
    axes[1, 0].grid()

    axes[0, 1].plot(t_B, y_B[0])
    axes[0, 1].set_title('Chamber B — Temperature')
    axes[0, 1].set_ylabel('T [K]')
    axes[0, 1].grid()

    axes[1, 1].plot(t_B, y_B[1])
    axes[1, 1].set_title('Chamber B — Pressure')
    axes[1, 1].set_ylabel('P [kPa]')
    axes[1, 1].set_xlabel('time [s]')
    axes[1, 1].grid()

    plt.show()
