"""Stand up -> wait for trigger -> slow crouch + recorded jump -> recover (numpy only).

Shared by the real-robot node (go2_jump_replay.py) and the MuJoCo check
(sim_check_jump_plan.py). SDK joint order: FR, FL, RR, RL x (hip, thigh, calf).

The trajectory file comes from extract_jump(): recorded samples from the crouch
start to t_req + END_AFTER_REQ, with the crouch-start / crouch-end indices.

Command = (q*, dq*, kp, kd, tau_ff); the motor driver applies
          tau = kp (q* - q) + kd (dq* - dq) + tau_ff.
"""

from __future__ import annotations

import json

import numpy as np

from crouch_plan import API_FRONTJUMP, find_phases

END_AFTER_REQ = 2.5        # replay this long after the original FrontJump request
SIT_POS = np.array([0.0, 1.4, -2.6] * 4)

STAND_KP, STAND_KD = 60.0, 5.0
JUMP_KP, JUMP_KD = 60.0, 3.0


def extract_jump(rec_npz, out_npz):
    d = np.load(rec_npz)
    meta = json.load(open(rec_npz.replace(".npz", ".json")))
    t_req = next(e["t"] for e in meta["events"]
                 if e["event"] == "request" and e["api_id"] == API_FRONTJUMP)
    t = d["t"]
    i0, i1 = find_phases(t, d["q"], t_req)
    i_end = int(np.searchsorted(t, t_req + END_AFTER_REQ))
    sl = slice(i0, i_end + 1)
    stand = (t > t_req - 0.5) & (t < t_req)
    hold = (t > t[i1] + 0.1) & (t < t_req + 1.0)
    np.savez(out_npz, t=t[sl] - t[i0], q=d["q"][sl], dq=d["dq"][sl], tau=d["tau_est"][sl],
             i_crouch_end=i1 - i0, tau_stand=d["tau_est"][stand].mean(0),
             tau_hold=d["tau_est"][hold].mean(0), source=rec_npz)
    return t[i_end] - t[i0], i1 - i0


class JumpPlan:
    """State machine driven by time and a trigger. Call step(t, trigger, sit)."""

    PREP, RECOVER, STAND_UP, SIT = 0.5, 1.5, 3.0, 2.0
    MIN_STAND = 2.0   # stand still this long before a (re)trigger is accepted

    def __init__(self, jump_npz, q_now, stretch=10.0, ff_scale=1.0):
        j = np.load(jump_npz)
        self.t, self.q, self.dq, self.tau = j["t"], j["q"], j["dq"], j["tau"] * ff_scale
        self.ic = int(j["i_crouch_end"])
        self.tc = float(self.t[self.ic])               # crouch length in the recording
        self.tau_stand = j["tau_stand"] * ff_scale
        self.tau_hold = j["tau_hold"] * ff_scale
        self.s = float(stretch)
        self.q_stance = self.q[0].copy()
        self.q_now = np.asarray(q_now, dtype=np.float64).copy()
        self.state, self.t_state = "hold", 0.0
        self.q_from = None
        self.jumps = 0

    # ── recorded trajectory ──
    def _rec(self, tr):
        tr = float(np.clip(tr, 0.0, self.t[-1]))
        i = int(np.clip(np.searchsorted(self.t, tr), 1, len(self.t) - 1))
        a = (tr - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        f = lambda x: (1 - a) * x[i - 1] + a * x[i]
        return f(self.q), f(self.dq), f(self.tau)

    def jump_duration(self):
        return self.tc * self.s + (self.t[-1] - self.tc)

    def _go(self, state, t):
        self.state, self.t_state = state, t

    def step(self, t, trigger=False, sit=False):
        """-> (state, q*, dq*, kp, kd, tau_ff)"""
        u = t - self.t_state
        z = np.zeros(12)
        smooth = lambda w: 0.5 - 0.5 * np.cos(np.pi * min(max(w, 0.0), 1.0))

        if sit and self.state == "standing":
            self.q_from = None
            self._go("sit", t); u = 0.0

        if self.state == "hold":                       # ramp gains on the current pose
            if u >= 1.0:
                self._go("stand_up", t); u = 0.0
            else:
                return "hold", self.q_now, z, STAND_KP * u, STAND_KD, z
        if self.state == "stand_up":
            w = smooth(u / self.STAND_UP)
            if u >= self.STAND_UP:
                self._go("standing", t); u = 0.0
            else:
                return "stand_up", self.q_now + w * (self.q_stance - self.q_now), z, STAND_KP, STAND_KD, z
        if self.state == "standing":
            if trigger and u >= self.MIN_STAND:
                self.jumps += 1
                self._go("prep", t); u = 0.0
            else:
                return "standing", self.q_stance, z, STAND_KP, STAND_KD, z
        if self.state == "prep":                        # blend gains and feed-forward in
            w = smooth(u / self.PREP)
            if u >= self.PREP:
                self._go("crouch", t); u = 0.0
            else:
                return "prep", self.q_stance, z, STAND_KP, STAND_KD + w * (JUMP_KD - STAND_KD), w * self.tau_stand
        if self.state == "crouch":                      # recorded crouch, slowed down
            tr = u / self.s
            if tr >= self.tc:
                self._go("jump", t); u = 0.0
            else:
                q, dq, _ = self._rec(tr)
                p = tr / self.tc
                ff = (1 - p) * self.tau_stand + p * self.tau_hold   # quasi-static
                return "crouch", q, dq / self.s, JUMP_KP, JUMP_KD, ff
        if self.state == "jump":                        # bottom hold, thrust, flight, landing
            tr = self.tc + u
            if tr >= self.t[-1]:
                self.q_from = self.q[-1].copy()
                self._go("recover", t); u = 0.0
            else:
                q, dq, tau = self._rec(tr)
                return "jump", q, dq, JUMP_KP, JUMP_KD, tau
        if self.state == "recover":                     # back to stance, ff and kd fade
            w = smooth(u / self.RECOVER)
            if u >= self.RECOVER:
                self._go("standing", t)
                return "standing", self.q_stance, z, STAND_KP, STAND_KD, z
            else:
                q = self.q_from + w * (self.q_stance - self.q_from)
                return "recover", q, z, STAND_KP, JUMP_KD + w * (STAND_KD - JUMP_KD), (1 - w) * self.tau[-1]
        if self.state == "sit":
            if self.q_from is None:
                self.q_from = self.q_stance.copy()
            w = smooth(u / self.SIT)
            if u >= self.SIT:
                return "done", SIT_POS, z, STAND_KP, STAND_KD, z
            return "sit", self.q_from + w * (SIT_POS - self.q_from), z, STAND_KP, STAND_KD, z
        return "done", SIT_POS, z, STAND_KP, STAND_KD, z
