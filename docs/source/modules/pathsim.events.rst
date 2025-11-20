Event Library
=============

PathSim's event system enables detection and handling of discrete events in hybrid dynamical systems.

----

Event Types
-----------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Zero-Crossing
      :link: pathsim.events.zerocrossing
      :link-type: doc

      Detect when a signal crosses zero or a threshold value for precise event timing.

   .. grid-item-card:: Scheduled Events
      :link: pathsim.events.schedule
      :link-type: doc

      Trigger events at predetermined times for periodic or timed system changes.

   .. grid-item-card:: Condition Events
      :link: pathsim.events.condition
      :link-type: doc

      Boolean condition-based events for complex triggering logic.

----

Base Classes
------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Event Base
      :link: pathsim.events._event
      :link-type: doc

      Base class for all event handlers with core functionality.

----

.. toctree::
   :hidden:
   :maxdepth: 1

   pathsim.events._event
   pathsim.events.condition
   pathsim.events.schedule
   pathsim.events.zerocrossing
