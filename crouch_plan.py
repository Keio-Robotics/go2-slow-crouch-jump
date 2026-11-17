"""Slow crouch-and-return plan built from a recorded FrontJump (numpy only).

Shared by the real-robot node (go2_crouch_replay.py) and the MuJoCo check
(sim_check_crouch_plan.py), so both run exactly the same targets.

Everything here is in Unitree SDK joint order: FR, FL, RR, RL x (hip, thigh, calf).

extract_crouch() keeps ONLY the crouch segment of the recording; the thrust,
flight and landing are dropped before anything is saved, so the file that goes
to the robot cannot command a jump.
"""

from __future__ import annotations

import json

import numpy as np

API_FRONTJUMP = 1031


def find_phases(t, q, t_req):
    calf = q[:, [2, 5, 8, 11]].mean(1)
    pre = calf[(t > t_req - 0.5) & (t < t_req)].mean()
    win = (t > t_req) & (t < t_req + 1.0)
    cmin = calf[win].min()
    i_end = np.where(win & (calf < cmin + 0.02))[0][0]
    i_start = np.where((t < t[i_end]) & (np.abs(calf - pre) < 0.03))[0][-1]
    return int(i_start), int(i_end)


def extract_crouch(rec_npz, out_npz):
    d = np.load(rec_npz)
    meta = json.load(open(rec_npz.replace(".npz", ".json")))
    t_req = next(e["t"] for e in meta["events"]
                 if e["event"] == "request" and e["api_id"] == API_FRONTJUMP)
    i0, i1 = find_phases(d["t"], d["q"], t_req)
    t = d["t"][i0:i1 + 1] - d["t"][i0]
    np.savez(out_npz, t=t, q=d["q"][i0:i1 + 1], source=rec_npz)
    return t[-1], d["q"][i0], d["q"][i1]


class CrouchPlan:
    """Targets as a function of time since the node started commanding.

    hold    gains ramp 0 -> kp on whatever pose the robot is in
    to_top  interpolate to the first sample of the crouch (crouch start pose)
    settle  stand still
    down    the recorded crouch, time-stretched by `stretch`
    bottom  hold the deepest pose
    up      the same crouch played backwards, same speed
    top     hold the start pose until the plan ends
    """

    def __init__(self, crouch_npz, q_now, stretch=10.0, kp=40.0,
                 hold=1.0, to_top=2.0, settle=1.0, bottom=1.0, top=2.0):
        c = np.load(crouch_npz)
        self.tc, self.qc = c["t"], c["q"]
        self.s, self.kp = float(stretch), float(kp)
        self.q_now = np.asarray(q_now, dtype=np.float64).copy()
        dur = self.tc[-1] * self.s
        self.phases = []
        t = 0.0
        for name, length in [("hold", hold), ("to_top", to_top), ("settle", settle),
                             ("down", dur), ("bottom", bottom), ("up", dur), ("top", top)]:
            self.phases.append((name, t, t + length))
            t += length
        self.T = t

    def _crouch(self, tau):  # tau: recording time within crouch
        tau = float(np.clip(tau, 0.0, self.tc[-1]))
        i = int(np.clip(np.searchsorted(self.tc, tau), 1, len(self.tc) - 1))
        a = (tau - self.tc[i - 1]) / max(self.tc[i] - self.tc[i - 1], 1e-9)
        q = (1 - a) * self.qc[i - 1] + a * self.qc[i]
        dq = (self.qc[i] - self.qc[i - 1]) / max(self.tc[i] - self.tc[i - 1], 1e-9)
        return q, dq

    def __call__(self, t):
        """-> (phase, q_target[12], dq_target[12], kp). phase 'done' after the end."""
        for name, a, b in self.phases:
            if t < b:
                u = t - a
                if name == "hold":
                    return name, self.q_now, np.zeros(12), self.kp * u / (b - a)
                if name == "to_top":
                    w = u / (b - a)
                    w = 0.5 - 0.5 * np.cos(np.pi * w)  # smooth start and stop
                    return name, self.q_now + w * (self.qc[0] - self.q_now), np.zeros(12), self.kp
                if name == "settle":
                    return name, self.qc[0], np.zeros(12), self.kp
                if name == "down":
                    q, dq = self._crouch(u / self.s)
                    return name, q, dq / self.s, self.kp
                if name == "bottom":
                    return name, self.qc[-1], np.zeros(12), self.kp
                if name == "up":
                    q, dq = self._crouch(self.tc[-1] - u / self.s)
                    return name, q, -dq / self.s, self.kp
                return name, self.qc[0], np.zeros(12), self.kp
        return "done", self.qc[0], np.zeros(12), self.kp
