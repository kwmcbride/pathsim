#########################################################################################
##
##                                 TIME SCHEDULED EVENTS
##                                  (events/schedule.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._event import Event

from .. _constants import TOLERANCE


# EVENT MANAGER CLASS ===================================================================

class Schedule(Event):
    """Subclass of base 'Event' that triggers dependent on the evaluation time. 
    
    Monitors time in every timestep and triggers periodically (period). This event
    does not have an event function as the event condition only depends on time.

    .. code-block::

        time == next_schedule_time -> event

    Example
    -------
    Initialize a scheduled event handler like this:

    .. code-block:: python

        #define the action function (callback)
        def act(t):
            #do something at event resolution
            pass
    
        #initialize the event manager
        E = Schedule(
            t_start=0,    #starting at t=0
            t_end=None,   #never ending
            t_period=3,   #triggering every 3 time units
            func_act=act  #resulting in a callback
            )   
    
    Parameters
    ----------
    t_start : float
        starting time for schedule
    t_end : float
        termination time for schedule
    t_period : float
        time period of schedule, when events are triggered
    func_act : callable
        action function for event resolution 
    tolerance : float
        tolerance to check if detection is close to actual event
    """

    def __init__(
        self, 
        t_start=0, 
        t_end=None, 
        t_period=1, 
        func_act=None,      
        tolerance=TOLERANCE
        ):
        super().__init__(None, func_act, tolerance)
        
        #schedule times
        self.t_start = t_start
        self.t_period = t_period        
        self.t_end = t_end


    def _next(self):
        """
        return the next period break
        """
        return self.t_start + len(self._times) * self.t_period


    def _spacing(self):
        """
        return the time between the upcoming event and its successor,
        used to bound the effective detection tolerance
        """
        return self.t_period


    def _tolerance_at(self, t):
        """Effective detection tolerance at evaluation time 't'.

        The simulation time is accumulated by repeated addition of the
        timestep, so it drifts away from the exact schedule times by a
        floating point error that grows with 't'. The configured absolute
        tolerance cannot cover this, which makes ticks that land within a
        few ulp of a step boundary flip between neighboring timesteps and
        lose the final tick of a run. The widened tolerance absorbs that
        drift while staying far below the schedule spacing, so no genuine
        tick can be swallowed.

        Parameters
        ----------
        t : float
            evaluation time for detection

        Returns
        -------
        float
            effective tolerance for the close-to-event check
        """
        return max(self.tolerance, min(1e-10 * abs(t), 0.1 * self._spacing()))


    def resolve(self, t):
        """Resolve the event at the exact scheduled time.

        The caller passes the numerically reached time, which carries the
        accumulated drift of the simulation clock. The schedule knows its
        own exact event time, so that is what gets recorded and handed to
        the action function - timestamps land exactly on the schedule.

        Parameters
        ----------
        t : float
            evaluation time for event resolution (ignored in favor of
            the exact scheduled time)
        """
        super().resolve(self._next())


    def estimate(self, t):
        """Estimate the time until the next scheduled event.

        Parameters
        ----------
        t : float 
            evaluation time for estimation 
        
        Returns
        -------
        float
            estimated time until next event
        """
        return self._next() - t


    def buffer(self, t):
        """Buffer the current time to history
        
        Parameters
        ----------
        t : float
            buffer time
        """
        self._history = None, t


    def detect(self, t):
        """Check if the event condition is satisfied, i.e. if the 
        time period switch is within the current timestep.
        
        Parameters
        ----------
        t : float
            evaluation time for detection 
        
        Returns
        -------
        detected : bool
            was an event detected?
        close : bool
            are we close to the event?
        ratio : float
            interpolated event location ratio in timestep
        """

        #get next period break
        t_next = self._next()

        #end time reached? -> deactivate event, quit early
        if self.t_end is not None and t_next > self.t_end:
            self.off()
            return False, False, 1.0

        #effective tolerance, covering the drift of the simulation clock
        _tol = self._tolerance_at(t)

        #no event -> quit early (a tick within tolerance of the step end
        #counts as inside the step, otherwise clock drift pushes it into
        #the next step and the final tick of a run is lost)
        if t_next > t + _tol:
            return False, False, 1.0

        #are we close enough to the scheduled event?
        if abs(t_next - t) <= _tol:
            return True, True, 1.0

        #unpack history
        _, _t = self._history

        #have we already passed the event -> first timestep
        if _t >= t_next:
            return True, True, 0.0

        #whats the timestep ratio?
        ratio = (t_next - _t) / np.clip(t - _t, TOLERANCE, None)

        return True, False, ratio


class ScheduleList(Schedule):
    """Subclass of base 'Schedule' that triggers dependent on the evaluation time. 
    
    Monitors time in every timestep and triggers at the next event time from the 
    time list. This event does not have an event function as the event condition 
    only depends on time.

    .. code-block::

        time == next_scheduled_time -> event

    Example
    -------
    Initialize a scheduled event handler like this:

    .. code-block:: python

        #define the action function (callback)
        def act(t):
            #do something at event resolution
            pass
    
        #initialize the event manager
        E = ScheduleList(
            times_evt=[1, 5, 12, 300],  #event times where to trigger
            func_act=act                #resulting in a callback
            )   
    
    Parameters
    ----------
    times_evt : list[float]
        list of event times in ascending order
    func_act : callable
        action function for event resolution 
    tolerance : float
        tolerance to check if detection is close to actual event
    """

    def __init__(
        self, 
        times_evt, 
        func_act=None,      
        tolerance=TOLERANCE
        ):
        super().__init__(t_start=None, func_act=func_act, tolerance=tolerance)

        #input validation for times
        if len(times_evt) > 1 and np.any(np.diff(times_evt) <= 0.0):
            raise ValueError("'times_evt' need to be in ascending order!")
        
        #schedule times
        self.times_evt = times_evt


    def _next(self):
        """return the next event from the event time list by index"""
        _n = len(self._times)
        if _n < len(self.times_evt):
            return self.times_evt[_n]
        return self.times_evt[-1]


    def _spacing(self):
        """return the gap between the upcoming event and its successor
        in the time list, used to bound the effective detection tolerance
        """
        _n = len(self._times)
        if _n + 1 < len(self.times_evt):
            return self.times_evt[_n + 1] - self.times_evt[_n]
        return float("inf")


    def detect(self, t):
        """Check if the event condition is satisfied, i.e. if the 
        time period switch is within the current timestep.
        
        Parameters
        ----------
        t : float
            evaluation time for detection 
        
        Returns
        -------
        detected : bool
            was an event detected?
        close : bool
            are we close to the event?
        ratio : float
            interpolated event location ratio in timestep
        """

        #check if out of bounds
        _n = len(self._times)
        if _n >= len(self.times_evt): 
            self.off()
            return False, False, 1.0

        #get next event time
        t_next = self._next()

        #effective tolerance, covering the drift of the simulation clock
        _tol = self._tolerance_at(t)

        #no event -> quit early (see 'Schedule.detect')
        if t_next > t + _tol:
            return False, False, 1.0

        #are we close enough to the scheduled event?
        if abs(t_next - t) <= _tol:
            return True, True, 1.0

        #unpack history
        _, _t = self._history

        #have we already passed the event -> first timestep
        if _t >= t_next:
            return True, True, 0.0

        #whats the timestep ratio?
        ratio = (t_next - _t) / np.clip(t - _t, TOLERANCE, None)

        return True, False, ratio
