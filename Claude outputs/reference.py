"""
Reference model

The commanded setpoint from the operator is a step.  Feeding that step
straight to the controller asks for a large control action in a single time
step, which saturates the thrusters and wastes actuator effort.  This module
shapes the commanded setpoint ``eta_cmd`` into a smooth, physically feasible
desired trajectory before the controller ever sees it.

Each controlled coordinate (N, E, psi) is passed through its own critically
damped second-order low-pass filter

    eta_d_ddot + 2*zeta*wn*eta_d_dot + wn^2*eta_d = wn^2*eta_cmd,

so a step in ``eta_cmd`` produces a position that leaves the current value
with zero initial velocity and acceleration, a bell-shaped velocity profile,
and no overshoot when ``zeta = 1``.  The filter states are exactly the
desired position, velocity and acceleration, so the same integration also
supplies the velocity and acceleration feedforward signals — no numerical
differentiation is needed.  An optional rate limit caps the desired speed so
that even very large setpoint jumps stay inside what the thrusters can do.

The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw].  The 3-DOF model uses indices [0, 1, 5]; the rest stay zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference position/heading
    nu_ref  : reference velocities   (d/dt of eta_ref)
    acc_ref : reference accelerations (d/dt of nu_ref)

The heading axis is filtered on the shortest angular error, so a command
across +-pi turns the short way instead of unwinding a full circle, and the
returned heading is wrapped to (-pi, pi].

All three arrays are always returned; a controller is free to use only the
position part and ignore the feedforward terms.
"""
from typing import Tuple

import numpy as np

# Per-axis tuning parameters live with the rest of the part_1 configuration.
from part_1.config import RefAxisConfig, default_ref_psi, default_ref_xy

_TWO_PI = 2.0 * np.pi

# Indices of the controlled coordinates in the 6-DOF vectors.
_IDX_N, _IDX_E, _IDX_PSI = 0, 1, 5


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((float(angle) + np.pi) % _TWO_PI - np.pi)


class ReferenceModel:
    """Critically damped second-order reference filter, one per axis.

    Parameters
    ----------
    dt :
        Nominal sample time [s] (only used if ``step`` is called without a
        usable ``dt``).
    cfg_xy, cfg_psi :
        Per-axis tuning (natural frequency, damping ratio, optional rate
        limit).  Defaults come from the configuration module so that the
        automated checks, which build ``ReferenceModel(dt)``, see the same
        tuning as the simulations.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else default_ref_xy()
        self.cfg_psi = cfg_psi if cfg_psi is not None else default_ref_psi()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state.

        Starting the filter at the measured state avoids an artificial jump
        at t = 0: the reference begins where the vessel actually is, at rest.
        """
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.eta_ref[_IDX_PSI] = _wrap_pi(self.eta_ref[_IDX_PSI])
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def _step_axis(
        self,
        x: float,
        v: float,
        x_cmd: float,
        cfg: RefAxisConfig,
        dt: float,
        is_angle: bool = False,
    ) -> Tuple[float, float, float]:
        """Advance one axis of the second-order filter by ``dt``.

        Semi-implicit (symplectic) Euler is used: the velocity is updated
        first and the new velocity drives the position.  It is one line of
        code, unconditionally stable for the sample times used here
        (wn*dt << 1), and it keeps position and velocity consistent, which
        matters because both are handed to the controller.
        """
        wn = float(cfg.wn)
        zeta = float(cfg.zeta)

        err = x_cmd - x
        if is_angle:
            err = _wrap_pi(err)          # always turn the short way

        acc = wn * wn * err - 2.0 * zeta * wn * v
        v_new = v + acc * dt

        # Optional saturation of the desired speed: a very large setpoint
        # jump then becomes a constant-speed transit with smooth ends
        # instead of a peak velocity the thrusters cannot deliver.
        limit = cfg.rate_limit
        if limit is not None and abs(v_new) > float(limit):
            v_new = float(np.sign(v_new) * float(limit))

        # Report the acceleration that actually produced v_new, so that
        # acc_ref is the exact derivative of nu_ref even while rate limited.
        acc_eff = (v_new - v) / dt
        x_new = x + v_new * dt
        if is_angle:
            x_new = _wrap_pi(x_new)
        return x_new, v_new, acc_eff

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the reference model one time step.

        Returns the desired position, velocity and acceleration as (6,)
        NED-frame arrays.
        """
        h = float(dt) if float(dt) > 0.0 else self.dt
        cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        eta = np.zeros(6)
        nu = np.zeros(6)
        acc = np.zeros(6)

        for idx, cfg, is_angle in (
            (_IDX_N, self.cfg_xy, False),
            (_IDX_E, self.cfg_xy, False),
            (_IDX_PSI, self.cfg_psi, True),
        ):
            eta[idx], nu[idx], acc[idx] = self._step_axis(
                self.eta_ref[idx], self.nu_ref[idx], cmd[idx], cfg, h, is_angle
            )

        self.eta_ref, self.nu_ref, self.acc_ref = eta, nu, acc
        return self.eta_ref, self.nu_ref, self.acc_ref
