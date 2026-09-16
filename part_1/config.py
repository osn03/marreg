# part_1/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 1 configuration — all tunable parameters in one place.

This file belongs to YOU, not to the simulator engine. Nothing in
``simulation/`` hardcodes a parameter you are asked to tune: the engine only
wires your models together, and every knob it needs is passed in from here
via ``run_case_part1.py``.

TODO (students): put your own configuration dataclasses in this file —
controller gains, thrust-allocation weights, wind/current parameters, ... —
so that every simulation can be reconfigured by editing this one file and
``run_case_part1.py``, without touching your model implementations. Example:

    @dataclass
    class PIDGains:
        Kp: np.ndarray = ...
        Ki: np.ndarray = ...
        Kd: np.ndarray = ...
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.thruster_dynamics import ThrusterConfig


@dataclass
class SimConfig:
    """Part 1 simulation clock and options (ideal thrusters by default)."""
    dt: float = 0.05
    T: float = 1000.0
    method: str = "Euler"
    use_reference: bool = True
    thruster_dynamics: bool = False  # Part 1: ideal actuators (no rate limits, no saturation)
    bypass_actuators: bool = False   # apply tau_d directly (debug)


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_1/reference.py).

    Each controlled coordinate is shaped by a second-order low-pass filter

        eta_d_ddot + 2*zeta*wn*eta_d_dot + wn**2 * eta_d = wn**2 * eta_cmd,

    so ``wn`` sets how fast the desired trajectory moves, ``zeta`` sets
    whether it overshoots, and ``rate_limit`` optionally caps the desired
    speed. The position axes (N, E) and the heading axis have very different
    inertia-to-authority ratios and are therefore tuned separately — see
    ``default_ref_xy()`` and ``default_ref_psi()`` below.

    Note: the automated checks build the default ``ReferenceModel(dt)``,
    which takes its per-axis tuning from those two factories — so keep the
    final tuned values there (overriding them only in ``run_case_part1.py``
    will not reach the checks).
    """
    wn: float = 0.05                    # natural frequency [rad/s]
    zeta: float = 1.0                   # damping ratio [-]
    rate_limit: Optional[float] = None  # max |x_dot| (m/s or rad/s); None = off


# -----------------------------------------------------------------------------
# Reference-model tuning (report: Reference Model section)
# -----------------------------------------------------------------------------
# The peak acceleration of a critically damped second-order filter occurs at
# the instant of the setpoint step and equals wn**2 * delta. The reference
# bandwidth is therefore not a free parameter: it is bounded by what the
# thrusters can actually deliver.
#
# Gunnerus zero-frequency inertia, rigid body + added mass
# (data/gunnerus_vessel.json):
#
#     m11 = 6.66e5 kg        m22 = 1.03e6 kg        m66 = 7.94e7 kg m^2
#
# Installed capacity from default_thrusters_gunnerus3():
#
#     surge   2 x 80 kN                       = 160 kN
#     sway    2 x 80 kN + 32 kN tunnel        = 192 kN
#     yaw     80 kN on a 13 m lever, twice    ~ 2000 kN m
#
# zeta = 1.0 on both axes. The desired trajectory must not overshoot the
# commanded setpoint: any overshoot in the reference is overshoot the
# thrusters have to create first and then undo.
#
# Position wn = 0.05 rad/s (T_r = 20 s). On the 50 m legs of Simulation 4
# this demands 0.125 m/s^2, i.e. 83 kN in surge (52% of capacity) and 129 kN
# in sway (67%). Sway is the binding axis, so the position bandwidth cannot
# be raised much: wn = 0.06 would need 186 kN in sway and is infeasible. A
# 50 m leg settles to within 0.5 m in 161 s with the rate limiter active
# (133 s without it), well inside the 300 s hold prescribed for the
# four-corner test; the residual error 50 s before the switch is 0.01 m.
#
# Heading wn = 0.10 rad/s (T_r = 10 s), twice the position bandwidth. Yaw is
# inertia-rich but also moment-rich, and the bandwidth that nearly saturates
# sway leaves yaw almost idle: the pi/4 turns of Simulation 4 need 623 kN m
# at this setting (about 31% of the available moment) and the pi/2 turn of
# Simulation 3 needs 1250 kN m (62%). Running heading at the position
# bandwidth would waste that authority and make the vessel crab through the
# four-corner turns instead of turning promptly onto the new heading.
# T_r = 10 s still leaves margin against the Part 2 azimuth swing time
# (0.2094 rad/s, i.e. 7.5 s for 90 deg).
#
# Rate limits cap the transit speed, so a large setpoint jump becomes a
# constant-speed transit with smooth ends instead of one short high-speed
# dash. The limiter binds only on the long moves:
#
#     position  0.5 m/s   binds for steps larger than 27 m
#                         (Simulation 4 legs yes, Simulation 3 step no)
#     heading   3 deg/s   binds for turns larger than 82 deg
#                         (Simulation 3 turn yes, Simulation 4 turns no)
#
# Because the reference model reports the acceleration that actually
# produced the new velocity, acc_ref remains the exact derivative of nu_ref
# while the limiter is active, so feedforward stays consistent.

REF_WN_XY: float = 0.05                  # position natural frequency [rad/s]
REF_WN_PSI: float = 0.10                 # heading natural frequency [rad/s]
REF_ZETA: float = 1.0                    # critical damping, both axes [-]
REF_RATE_XY: float = 0.5                 # max desired speed [m/s]
REF_RATE_PSI: float = np.deg2rad(3.0)    # max desired yaw rate [rad/s]


def default_ref_xy() -> RefAxisConfig:
    """Tuned reference-model settings for the position axes (N and E)."""
    return RefAxisConfig(wn=REF_WN_XY, zeta=REF_ZETA, rate_limit=REF_RATE_XY)


def default_ref_psi() -> RefAxisConfig:
    """Tuned reference-model settings for the heading axis (psi)."""
    return RefAxisConfig(wn=REF_WN_PSI, zeta=REF_ZETA, rate_limit=REF_RATE_PSI)


def default_thrusters_gunnerus3() -> list[ThrusterConfig]:
    """Three-thruster Gunnerus layout from the project description (Table 3)."""
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel",  x=+12.0, y=0.0,
                       u_max=32000,  u_rate=4000,  rot_speed=0.0,    alpha0=np.pi / 2),
        ThrusterConfig("Azimuth_1",  "azimuth", x=-13.0, y=+3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
        ThrusterConfig("Azimuth_2",  "azimuth", x=-13.0, y=-3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
    ]
