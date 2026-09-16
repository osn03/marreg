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

    The reference model is a critically damped second-order low-pass filter
    per controlled coordinate:

        eta_d_ddot + 2*zeta*wn*eta_d_dot + wn^2*eta_d = wn^2*eta_cmd.

    Note: the automated checks build the default ``ReferenceModel(dt)``, which
    uses ``default_ref_xy()`` / ``default_ref_psi()`` below — so keep your
    final tuned values here (overriding them only in ``run_case_part1.py`` will not
    reach the checks).

    Tuning rationale (report, Reference Model section):

    * ``zeta = 1`` (critical damping) — the desired trajectory must not
      overshoot the commanded setpoint; any overshoot in the reference is
      overshoot the thrusters have to create and then undo.
    * ``wn = 0.05 rad/s`` for the position axes — time constant
      T_r = 1/wn = 20 s, i.e. a 10 m step is essentially completed in about
      100 s with a peak desired speed of A*wn/e = 0.18 m/s.  This is slow
      compared with the actuators: the azimuths need ~8 s to ramp from zero
      to full thrust (10 kN/s up to 80 kN) and ~7.5 s to swing 90 deg
      (0.2094 rad/s), so with T_r = 20 s the allocation never has to fight
      its own rate limits, and the thrusters stay far from saturation.
    * ``wn = 0.08 rad/s`` for heading — T_r = 12.5 s.  Yaw can be driven
      faster than translation because the two aft azimuths give a large yaw
      moment at short arm cost, and heading changes do not have to move the
      whole added mass of the hull.
    * ``rate_limit`` caps the desired speed so that a very large setpoint
      jump becomes a constant-speed transit instead of an infeasible peak:
      0.5 m/s for position, 3 deg/s for heading.  For the step sizes used in
      the mandatory simulations the limit is not active — it only protects
      against operator commands far outside them.
    """
    wn: float = 0.05                    # natural frequency [rad/s] (position axes)
    zeta: float = 1.0                   # damping ratio [-] (critically damped)
    rate_limit: Optional[float] = 0.5   # max |x_dot| [m/s]; None = off


def default_ref_xy() -> RefAxisConfig:
    """Tuned reference-model parameters for the N and E axes."""
    return RefAxisConfig(wn=0.05, zeta=1.0, rate_limit=0.5)


def default_ref_psi() -> RefAxisConfig:
    """Tuned reference-model parameters for the heading axis."""
    return RefAxisConfig(wn=0.08, zeta=1.0, rate_limit=float(np.deg2rad(3.0)))


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
