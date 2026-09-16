# part_2/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 2 configuration — all tunable parameters in one place.

Like ``part_1/config.py``, this file belongs to you: put your own
configuration dataclasses here (observer parameters, wave-model settings,
controller gains, ...) so that every Part 2 simulation can be reconfigured
from this file and ``run_case_part_2.py`` without touching the engine.
"""
from dataclasses import dataclass, field
from math import pi
from typing import Any, Dict, Optional

from models.thruster_dynamics import ThrusterConfig


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_2/reference.py).

    The reference model is a critically damped second-order low-pass filter
    per controlled coordinate:

        eta_d_ddot + 2*zeta*wn*eta_d_dot + wn^2*eta_d = wn^2*eta_cmd.

    Note: the automated checks build the default ``ReferenceModel(dt)``, which
    uses ``default_ref_xy()`` / ``default_ref_psi()`` below — so keep your
    final tuned values here (overriding them only in ``run_case_part_2.py`` will not
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
    return RefAxisConfig(wn=0.08, zeta=1.0, rate_limit=3.0 * pi / 180.0)


@dataclass
class Part2SimConfig:
    """Part 2-only clock and switches; it does not inherit Part 1 settings.

    ``use_sensor_noise`` switches measurement noise on for the optional
    sensor-noise part of the project.  The noise levels are fixed course
    parameters in ``models/sensors.py`` — the switch is the only sensor
    setting you select per simulation.  The mandatory simulations run with
    ideal measurements (``False``, the default).

    ``use_controller=False`` switches the DP system off: the controller is
    not called, the desired wrench is zero and the vessel drifts freely under
    the environmental loads (Simulation 1).  ``use_observer`` still works in
    that mode, so an observer can be watched on a drifting vessel.

    ``dt`` must be 0.1 or 0.01 s (the engine rejects other steps; the Part 1
    default of 0.05 s does not carry over).  ``observer_kwargs`` are passed to
    ``select_observer(observer_type, **observer_kwargs)`` when the engine
    builds the observer, so a differently tuned observer can be selected from
    the configuration (e.g. ``{"cfg": MyObserverConfig(...)}``).
    """
    dt: float = 0.1
    T: float = 1000.0
    method: str = "Euler"
    use_controller: bool = True
    use_reference: bool = True
    use_observer: bool = False
    # Which observer the engine builds when none is injected.  This is only a
    # plumbing default so the field is never empty, NOT a recommendation:
    # every preset that needs an observer names the type explicitly, and which
    # one you adopt is your Part 2, Simulation 4 decision.
    observer_type: str = "nonlinear_passive"
    use_sensor_noise: bool = False
    thruster_dynamics: bool = True
    bypass_actuators: bool = False
    observer_kwargs: Dict[str, Any] = field(default_factory=dict)


def default_thrusters_part_2() -> list[ThrusterConfig]:
    """Same Gunnerus thruster fit as Project Part 1.

    Part 2 extends Part 1: the vessel and its three thrusters are unchanged.
    What is new in Part 2 is that the thruster *constraints* (max thrust,
    ramp rate ``u_rate`` and azimuth rotation speed) are enforced by the
    thruster dynamics and must be respected by your thrust allocation.
    The azimuth rotation limit is 2 rpm = 0.20944 rad/s.  Angles are in the
    BODY frame, where the fixed tunnel thruster points at ``+pi/2``.
    """
    rotation_speed = 2.0 * 2.0 * pi / 60.0
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel", x=12.0, y=0.0,
                       u_max=32_000.0, u_rate=4_000.0,
                       rot_speed=0.0, alpha0=pi / 2.0),
        ThrusterConfig("Azimuth_1", "azimuth", x=-13.0, y=3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
        ThrusterConfig("Azimuth_2", "azimuth", x=-13.0, y=-3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
    ]
