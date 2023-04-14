from typing import Sequence, Union, List
import math
import abc
from xmlrpc.client import Boolean
from numba.experimental import jitclass
from numba import jit, types, typed, typeof
import typing as pt
import numba

float_type = types.float32
@jitclass
class RampSpeedProfile:
    """
    Accelerate early and cruise if reached end_speed (i.e. max_speed).
    If not possible, keep distance and acceleration, compromise end_speed.
    """
    is_acc: Boolean
    d_acc: float_type
    t_acc: float_type
    d_cruise: float_type
    t_cruise: float_type
    start_speed: float_type
    end_speed: float_type
    acceleration: float_type
    distance: float_type
    duration: float_type

    def __init__(self, 
            distance, 
            start_speed=0, 
            end_speed=1, 
            acceleration=1,
            strict=False):
        assert(distance >= 0)
        assert(start_speed >= 0)
        assert(end_speed >= 0)
        assert(acceleration >= 0)

        is_acc = start_speed <= end_speed
        if is_acc:
            max_end_speed = math.sqrt(2*acceleration*distance+start_speed*start_speed)
            assert(max_end_speed >= start_speed)
            if strict:
                raise RuntimeError('Insufficient acceleartion.')
            end_speed = min(max_end_speed, end_speed)

            t_acc = (end_speed - start_speed) / acceleration
            d_acc = start_speed * t_acc + 0.5 * acceleration * t_acc * t_acc
            d_cruise = distance - d_acc
            t_cruise = d_cruise / end_speed
        else:
            acceleration = -acceleration
            min_t_acc = start_speed / abs(acceleration)
            pure_d_acc = min_t_acc * start_speed / 2
            # if can decelerate to 0 within distance
            # have sufficient acceleration, not need to change end_speed
            if pure_d_acc > distance:
                min_end_speed = math.sqrt(2*acceleration*distance+start_speed*start_speed)
                assert(min_end_speed <= start_speed)
                if strict:
                    raise RuntimeError('Insufficient acceleartion.')
                end_speed = max(min_end_speed, end_speed)

            t_acc = (end_speed - start_speed) / acceleration
            d_acc = start_speed * t_acc + 0.5 * acceleration * t_acc * t_acc
            d_cruise = distance - d_acc
            t_cruise = d_cruise / start_speed
        
        assert(abs(end_speed - start_speed - t_acc * acceleration) < 1e-7)


        self.is_acc = is_acc
        self.d_acc = d_acc
        self.t_acc = t_acc
        self.d_cruise = d_cruise
        self.t_cruise = t_cruise

        self.start_speed = start_speed
        self.end_speed = end_speed
        self.acceleration = acceleration
        self.distance = distance
        # total time
        self.duration = self.t_cruise + self.t_acc
    
    def ddq(self, t):
        if self.is_acc:
            if t <= self.t_acc:
                return self.acceleration
            else:
                return 0
        else:
            if t < self.t_cruise:
                return 0
            else:
                return self.acceleration

    def dq(self, t):
        if self.is_acc:
            if t <= self.t_acc:
                return self.acceleration * t + self.start_speed
            else:
                return self.end_speed
        else:
            if t < self.t_cruise:
                return self.start_speed
            else:
                return self.start_speed + (t - self.t_cruise) * self.acceleration

    def q(self, t):
        if self.is_acc:
            if t <= self.t_acc:
                return self.start_speed * t + self.acceleration * t * t / 2
            else:
                return self.d_acc + (t - self.t_acc) * self.end_speed
        else:
            if t < self.t_cruise:
                return self.start_speed * t
            else:
                ta = (t - self.t_cruise)
                return self.d_cruise + self.start_speed * ta + 0.5 * self.acceleration * ta * ta

@jitclass
class TrapezoidalSpeedProfile:
    """
    Cruise and decelerate in th end.
    If not possible, keep distance and acceleration, compromise end_speed.
    """
    d_acc: float_type
    t_acc: float_type
    d_cruise: float_type
    t_cruise: float_type
    start_speed: float_type
    end_speed: float_type
    max_speed: float_type
    acceleration: float_type
    distance: float_type
    duration: float_type

    def __init__(self, 
            distance, 
            speed=1,
            acceleration=1):
        assert(distance >= 0)
        assert(speed > 0)
        assert(acceleration > 0)

        t_cruise = None
        t_acc = None
        max_speed = None
        tri_max_speed = math.sqrt(acceleration * distance)
        if tri_max_speed <= speed:
            # triangle
            t_acc = distance / tri_max_speed
            t_cruise = 0
            max_speed = tri_max_speed
        else:
            # trapozoid
            t_acc = speed / acceleration
            tri_travel = t_acc * speed
            t_cruise = (distance - tri_travel) / speed
            max_speed = speed

        duration = t_acc * 2 + t_cruise

        self.t_acc = t_acc
        self.t_cruise = t_cruise
        self.d_acc = t_acc * max_speed / 2
        self.d_cruise = t_cruise * max_speed
        self.max_speed = max_speed
        self.distance = distance
        self.duration = duration
        self.acceleration = acceleration
        self.start_speed = 0
        self.end_speed = 0
    
    def ddq(self, t):
        if t <= self.t_acc:
            return self.acceleration
        elif t >= (self.t_acc + self.t_cruise):
            return -self.acceleration
        else:
            return 0

    def dq(self, t):
        if t <= self.t_acc:
            return t * self.acceleration
        elif t >= (self.t_acc + self.t_cruise):
            ta = t - self.t_acc - self.t_cruise
            return self.max_speed - ta * self.acceleration
        else:
            return self.max_speed

    def q(self, t):
        if t <= self.t_acc:
            return 0.5 * self.acceleration * t * t
        elif t >= (self.t_acc + self.t_cruise):
            ta = t - self.t_acc - self.t_cruise
            return self.d_acc + self.d_cruise \
                + self.max_speed * ta \
                - 0.5 * self.acceleration * ta * ta
        else:
            tc = t - self.t_acc
            return self.d_acc + self.max_speed * tc

@jitclass
class SpeedProfileInterval:
    begin: float_type
    end: float_type
    left_distance: float_type
    speed_profile: RampSpeedProfile

    def __init__(self, begin, end, left_distance, speed_profile):
        self.begin = begin
        self.end = end
        self.left_distance = left_distance
        self.speed_profile = speed_profile

@jitclass
class SequentialSpeedProfile:
    """
    A sequence of speed profiles.
    Requirement:
    * start/end speed must be equal
    """
    start_speed: float_type
    end_speed: float_type
    distance: float_type
    duration: float_type
    intervals: pt.List[SpeedProfileInterval]

    def __init__(self, start_speed, end_speed, total_distance, total_duration, intervals):
        self.start_speed = start_speed
        self.end_speed = end_speed
        self.distance = total_distance
        self.duration = total_duration
        self.intervals = intervals
    
    def find_interval(self, t):
        left_idx = 0
        right_idx = len(self.intervals) - 1
        while left_idx != right_idx:
            mid_idx = (left_idx + right_idx) // 2
            if t < self.intervals[mid_idx].begin:
                right_idx = mid_idx - 1
            elif t >= self.intervals[mid_idx].end:
                left_idx = mid_idx + 1
            else:
                return self.intervals[mid_idx]
        return self.intervals[left_idx]
    
    def ddq(self, t):
        interval = self.find_interval(t)
        st = t - interval.begin
        return interval.speed_profile.ddq(st)
    
    def dq(self, t):
        interval = self.find_interval(t)
        st = t - interval.begin
        return interval.speed_profile.dq(st)

    def q(self, t):
        interval = self.find_interval(t)
        st = t - interval.begin
        return interval.left_distance + interval.speed_profile.q(st)

def sequential_speed_profile(segments: List[RampSpeedProfile]):
    total_duration = 0
    total_distance = 0
    intervals = typed.List.empty_list(SpeedProfileInterval.class_type.instance_type)
    for segment in segments:
        if len(intervals) > 0:
            assert(abs(intervals[-1].speed_profile.end_speed - segment.start_speed) < 1e-7)

        # left = -math.inf if len(intervals) == 0 else total_duration
        left = total_duration
        right = math.inf if \
            len(intervals) == (len(segments) - 1) else total_duration + segment.duration
        interval = SpeedProfileInterval(
            left, right, total_distance, segment)
        intervals.append(interval)
        total_duration += segment.duration
        total_distance += segment.distance

    start_speed = segments[0].start_speed
    end_speed = segments[-1].end_speed
    seq_profile = SequentialSpeedProfile(
        start_speed, end_speed, 
        total_distance, total_duration, 
        intervals)
    return seq_profile


# %%
def test():
    from matplotlib import pyplot as plt
    import numpy as np

    sp = RampSpeedProfile(1, 1, 2, 10)
    t = np.linspace(0, sp.duration, 100)
    q = np.array([sp.q(x) for x in t])
    dq = np.array([sp.dq(x) for x in t])
    ddq = np.array([sp.ddq(x) for x in t])

    sp = RampSpeedProfile(1, 2, 0, 0.1)
    t = np.linspace(0, sp.duration, 100)
    q = np.array([sp.q(x) for x in t])
    dq = np.array([sp.dq(x) for x in t])
    ddq = np.array([sp.ddq(x) for x in t])

    sp = TrapezoidalSpeedProfile(1,1,1)
    t = np.linspace(0, sp.duration, 100)
    q = np.array([sp.q(x) for x in t])
    dq = np.array([sp.dq(x) for x in t])
    ddq = np.array([sp.ddq(x) for x in t])

    segments = [RampSpeedProfile(1,0,1,1), RampSpeedProfile(1,1,0,1)]
    sp = sequential_speed_profile(segments)

    t = np.linspace(0, sp.duration, 100)
    q = np.array([sp.q(x) for x in t])
    dq = np.array([sp.dq(x) for x in t])
    ddq = np.array([sp.ddq(x) for x in t])
 

