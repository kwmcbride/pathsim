#########################################################################################
##
##                    PathSim BusMerge Example: HVAC Zone Monitoring
##
##  Demonstrates Bus definition reuse, nested BusCreator, BusMerge, and
##  bus-aware Scope with a nested bus schema.
##
##  Models two independently-controlled HVAC zones (A and B) that share the
##  same typed bus definition, plus an outdoor weather station.
##
##  Because both zones share one schema (same key names), they cannot be
##  merged directly — that would cause key conflicts.  Instead they are first
##  namespaced into a nested zones-bundle via a BusCreator(['ZoneA','ZoneB']),
##  and then BusMerge combines that bundle with the outdoor bus:
##
##      zone_A_bus  ─┐
##                   ├─ BusCreator(['ZoneA','ZoneB']) ─┐
##      zone_B_bus  ─┘                                  ├─ BusMerge ─ Scope
##                                                      │
##      outdoor_bus ───────────────────────────────────-┘
##
##  Zone dynamics — first-order thermal model:
##
##      dT/dt = k * (T_setpoint - T)
##
##    Zone A: T0 = 16 C,  setpoint = 21 C,  k = 0.05 s⁻¹
##    Zone B: T0 = 28 C,  setpoint = 22 C,  k = 0.03 s⁻¹
##
##                               Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Integrator, Amplifier, Adder, Constant, Scope,
    BusCreator, BusMerge,
)
from pathsim.bus import Bus, BusElement
from pathsim.solvers import SSPRK22


# BUS DEFINITIONS =======================================================================
#
# zone_bus_def is defined once and reused for both zones — they share
# the same signal schema, just carry different values at runtime.

zone_bus_def = Bus('ZoneBus', elements=[
    BusElement('Temperature', data_type='float', unit='C',   description='Zone air temperature'),
    BusElement('Humidity',    data_type='float', unit='%RH', description='Zone relative humidity'),
], description='Sensor readings for one HVAC zone')

outdoor_bus_def = Bus('Outdoor', elements=[
    BusElement('T_outdoor',  data_type='float', unit='C',   description='Outdoor temperature'),
    BusElement('WindSpeed',  data_type='float', unit='m/s', description='Wind speed'),
], description='Outdoor weather station')

# System monitoring bus.
# The two zones are nested under 'ZoneA' and 'ZoneB' — each of type zone_bus_def.
# Outdoor signals sit at the top level alongside the zone sub-buses.
# Scope(bus=system_bus_def) will automatically derive dot-path labels:
#   'ZoneA.Temperature [C]', 'ZoneA.Humidity [%RH]', 'ZoneB...', 'T_outdoor...', ...
system_bus_def = Bus('SystemBus', elements=[
    BusElement('ZoneA',     data_type=zone_bus_def),
    BusElement('ZoneB',     data_type=zone_bus_def),
    BusElement('T_outdoor', data_type='float', unit='C'),
    BusElement('WindSpeed', data_type='float', unit='m/s'),
], description='Full HVAC system monitoring bus')


# ZONE A — thermal dynamics =============================================================
#
#   dT_A/dt = k_A * T_set_A  -  k_A * T_A
#           = set_A_const    +  amp_fb_A(T_A)
#
#   Both zones use zone_bus_def, so their BusCreators are configured from
#   the same definition object.

T0_A    = 16.0    # C   — initial temperature
T_set_A = 21.0    # C   — target setpoint
k_A     = 0.05    # 1/s

set_A    = Constant(k_A * T_set_A)   # constant feed-forward term
T_A_int  = Integrator(T0_A)
fb_A     = Amplifier(-k_A)           # negative feedback
dTdt_A   = Adder('++')               # dT_A/dt = set_A_const + fb_A(T_A)
hum_A    = Constant(55.0)            # %RH

creator_A = BusCreator(keys=zone_bus_def)   # reuses shared schema


# ZONE B — thermal dynamics =============================================================

T0_B    = 28.0
T_set_B = 22.0
k_B     = 0.03

set_B    = Constant(k_B * T_set_B)
T_B_int  = Integrator(T0_B)
fb_B     = Amplifier(-k_B)
dTdt_B   = Adder('++')
hum_B    = Constant(42.0)

creator_B = BusCreator(keys=zone_bus_def)   # same definition, different instance


# OUTDOOR WEATHER STATION ===============================================================

T_out = Constant(8.0)    # C
wind  = Constant(3.5)    # m/s

creator_out = BusCreator(keys=outdoor_bus_def)


# NAMESPACE + MERGE =====================================================================
#
# zones_creator wraps the two same-schema zone buses under distinct string
# keys before merging, avoiding the key conflict that would arise from
# merging them directly.

zones_creator = BusCreator(keys=['ZoneA', 'ZoneB'])   # accepts bus dicts as inputs

# BusMerge combines the zones bundle (nested dict) with the flat outdoor bus.
merger = BusMerge(n=2)

# Scope receives the merged nested bus and records all six leaf channels.
scope = Scope(bus=system_bus_def)


# BLOCKS + CONNECTIONS ==================================================================

blocks = [
    set_A, T_A_int, fb_A, dTdt_A, hum_A, creator_A,
    set_B, T_B_int, fb_B, dTdt_B, hum_B, creator_B,
    T_out, wind, creator_out,
    zones_creator, merger, scope,
]

connections = [
    # Zone A dynamics
    Connection(T_A_int[0], fb_A[0]),
    Connection(set_A[0],   dTdt_A[0]),
    Connection(fb_A[0],    dTdt_A[1]),
    Connection(dTdt_A[0],  T_A_int[0]),

    # Zone A bus: Temperature and Humidity → creator_A
    Connection(T_A_int[0], creator_A['Temperature']),
    Connection(hum_A[0],   creator_A['Humidity']),

    # Zone B dynamics
    Connection(T_B_int[0], fb_B[0]),
    Connection(set_B[0],   dTdt_B[0]),
    Connection(fb_B[0],    dTdt_B[1]),
    Connection(dTdt_B[0],  T_B_int[0]),

    # Zone B bus
    Connection(T_B_int[0], creator_B['Temperature']),
    Connection(hum_B[0],   creator_B['Humidity']),

    # Outdoor bus
    Connection(T_out[0], creator_out['T_outdoor']),
    Connection(wind[0],  creator_out['WindSpeed']),

    # Namespace the two zone buses into a nested dict
    Connection(creator_A[0], zones_creator['ZoneA']),
    Connection(creator_B[0], zones_creator['ZoneB']),

    # Merge zones bundle + outdoor into the system bus
    Connection(zones_creator[0], merger['bus_0']),
    Connection(creator_out[0],   merger['bus_1']),

    # Record merged bus directly — Scope expands nested paths automatically
    Connection(merger[0], scope[0]),
]


# SIMULATION ============================================================================

sim = Simulation(blocks, connections, Solver=SSPRK22, dt=0.5)


# RUN + PLOT ============================================================================

if __name__ == '__main__':

    sim.run(duration=120.0)

    t, y = scope.read()

    # Labels derived from system_bus_def:
    #   'ZoneA.Temperature [C]', 'ZoneA.Humidity [%RH]',
    #   'ZoneB.Temperature [C]', 'ZoneB.Humidity [%RH]',
    #   'T_outdoor [C]', 'WindSpeed [m/s]'
    labels = scope.labels

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), tight_layout=True)

    for idx, ax in enumerate(axes.flat):
        ax.plot(t, y[idx])
        ax.set_title(labels[idx])
        ax.set_xlabel('time [s]')
        ax.grid(True)

    fig.suptitle('HVAC System Monitoring — nested bus + BusMerge', fontsize=12)
    plt.show()
