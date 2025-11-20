Solver Library
==============

PathSim provides a comprehensive suite of numerical integrators for solving ordinary differential equations.

----

Explicit Runge-Kutta Methods
-----------------------------

Fast, non-iterative solvers ideal for non-stiff problems.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Euler
      :link: pathsim.solvers.euler
      :link-type: doc

      First-order explicit forward Euler method, simplest integrator for basic problems.

   .. grid-item-card:: RK4
      :link: pathsim.solvers.rk4
      :link-type: doc

      Classic fourth-order Runge-Kutta method with excellent accuracy-to-cost ratio.

   .. grid-item-card:: SSPRK22
      :link: pathsim.solvers.ssprk22
      :link-type: doc

      Strong Stability Preserving RK method, 2nd order with 2 stages.

   .. grid-item-card:: SSPRK33
      :link: pathsim.solvers.ssprk33
      :link-type: doc

      Strong Stability Preserving RK method, 3rd order with 3 stages.

   .. grid-item-card:: SSPRK34
      :link: pathsim.solvers.ssprk34
      :link-type: doc

      Strong Stability Preserving RK method, 3rd order with 4 stages.

----

Adaptive Runge-Kutta Methods
-----------------------------

Embedded methods with automatic step-size control for efficient integration.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: RKF21
      :link: pathsim.solvers.rkf21
      :link-type: doc

      Fehlberg's 2nd/1st order adaptive method for simple non-stiff problems.

   .. grid-item-card:: RKBS32 
      :link: pathsim.solvers.rkbs32
      :link-type: doc

      3rd/2nd order adaptive method from Bogacki and Shampine.

   .. grid-item-card:: RKF45 
      :link: pathsim.solvers.rkf45
      :link-type: doc

      4th/5th order adaptive method (Fehlberg), widely used classic solver.

   .. grid-item-card:: RKCK54 
      :link: pathsim.solvers.rkck54
      :link-type: doc

      5th/4th order adaptive method (Cash-Karp) with optimized error coefficients.

   .. grid-item-card:: RKDP54 
      :link: pathsim.solvers.rkdp54
      :link-type: doc

      5th/4th order adaptive method (Dormand-Prince), often the default choice for non-stiff problems.

   .. grid-item-card:: RKV65
      :link: pathsim.solvers.rkv65
      :link-type: doc

      Verner's 6th/5th order adaptive method for high-accuracy requirements.

   .. grid-item-card:: RKF78
      :link: pathsim.solvers.rkf78
      :link-type: doc

      7th/8th order adaptive method (Fehlberg) for very high precision applications.

   .. grid-item-card:: RKDP87 
      :link: pathsim.solvers.rkdp87
      :link-type: doc

      8th/7th order adaptive method (Dormand-Prince) for extreme accuracy demands.

----

Implicit Methods
----------------

Iterative solvers for stiff differential equations and algebraic-differential systems.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: BDF
      :link: pathsim.solvers.bdf
      :link-type: doc

      Backward Differentiation Formulas (fixed step) for stiff problems with strong stability.

   .. grid-item-card:: GEAR
      :link: pathsim.solvers.gear
      :link-type: doc

      Gear's method for stiff differential equations, adaptive timestepping variants of BDF.

   .. grid-item-card:: DIRK2
      :link: pathsim.solvers.dirk2
      :link-type: doc

      2nd order Diagonally Implicit Runge-Kutta method, A-stable and SSP-optimal.

   .. grid-item-card:: DIRK3
      :link: pathsim.solvers.dirk3
      :link-type: doc

      3rd order Diagonally Implicit Runge-Kutta method, L-stable.

   .. grid-item-card:: ESDIRK32
      :link: pathsim.solvers.esdirk32
      :link-type: doc

      Explicit first stage DIRK, 3rd/2nd order adaptive method.

   .. grid-item-card:: ESDIRK4
      :link: pathsim.solvers.esdirk4
      :link-type: doc

      Explicit first stage DIRK, 4th order for stiff problems.

   .. grid-item-card:: ESDIRK43
      :link: pathsim.solvers.esdirk43
      :link-type: doc

      Explicit first stage DIRK, 4th/3rd order adaptive method.

   .. grid-item-card:: ESDIRK54
      :link: pathsim.solvers.esdirk54
      :link-type: doc

      Explicit first stage DIRK, 5th/4th order adaptive high-accuracy solver.

   .. grid-item-card:: ESDIRK85
      :link: pathsim.solvers.esdirk85
      :link-type: doc

      Explicit first stage DIRK, 8th/5th order adaptive for very high precision.

----

Special Solvers
---------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Steady State
      :link: pathsim.solvers.steadystate
      :link-type: doc

      Time-independent steady-state solver for finding DC operating points and equilibria.

----

Base Classes
------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Solver Base
      :link: pathsim.solvers._solver
      :link-type: doc

      Base class for all numerical integrators with core integration functionality.

   .. grid-item-card:: Runge-Kutta Base
      :link: pathsim.solvers._rungekutta
      :link-type: doc

      Base class for Runge-Kutta family methods with tableau-based implementation.

----

.. toctree::
   :hidden:
   :maxdepth: 1

   pathsim.solvers._solver
   pathsim.solvers._rungekutta
   pathsim.solvers.euler
   pathsim.solvers.ssprk22
   pathsim.solvers.ssprk33
   pathsim.solvers.ssprk34
   pathsim.solvers.rk4
   pathsim.solvers.rkbs32
   pathsim.solvers.rkck54
   pathsim.solvers.rkdp54
   pathsim.solvers.rkdp87
   pathsim.solvers.rkf21
   pathsim.solvers.rkf45
   pathsim.solvers.rkf78
   pathsim.solvers.rkv65
   pathsim.solvers.bdf
   pathsim.solvers.gear
   pathsim.solvers.dirk2
   pathsim.solvers.dirk3
   pathsim.solvers.esdirk32
   pathsim.solvers.esdirk4
   pathsim.solvers.esdirk43
   pathsim.solvers.esdirk54
   pathsim.solvers.esdirk85
   pathsim.solvers.steadystate
