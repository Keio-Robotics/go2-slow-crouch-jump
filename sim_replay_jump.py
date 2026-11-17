#!/usr/bin/env python3
"""Replay a recorded real-Go2 FrontJump (go2_record_jump.py output) in MuJoCo.

Each joint is driven by  tau = kp*(q_ref - q) + kd*(dq_ref - dq) + ff*tau_ref,
clipped to the actuator limits. The crouch segment of the recording can be
time-stretched by --stretch; everything after it is replayed unchanged.

    python3 sim_replay_jump.py data/jump01.npz                # as recorded
    python3 sim_replay_jump.py data/jump01.npz --stretch 10   # 10x slower crouch
    ... --video out.mp4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# MuJoCo Menagerie Go2 scene; clone https://github.com/google-deepmind/mujoco_menagerie
# next to this file, or point GO2_SCENE at scene.xml.
SCENE = os.environ.get("GO2_SCENE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_go2", "scene.xml"))
# SDK order FR,FL,RR,RL  ->  MuJoCo model order FL,FR,RL,RR
SDK_TO_MODEL = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def find_phases(d, t_req):
    """Return (crouch_start, crouch_end) in recording time.

    crouch_end = first instant the mean calf angle is within 0.02 rad of its
    minimum during the jump; crouch_start = last instant before that at which
    it was still within 0.03 rad of the pre-jump stance."""
    t, q = d["t"], d["q"]
    calf = q[:, [2, 5, 8, 11]].mean(1)
    pre = calf[(t > t_req - 0.5) & (t < t_req)].mean()
    win = (t > t_req) & (t < t_req + 1.0)
    cmin = calf[win].min()
    i_end = np.where(win & (calf < cmin + 0.02))[0][0]
    i_start = np.where((t < t[i_end]) & (np.abs(calf - pre) < 0.03))[0][-1]
    return float(t[i_start]), float(t[i_end])


class Reference:
    """Recorded trajectory resampled on a warped clock."""

    def __init__(self, d, t0, c0, c1, stretch):
        self.t = d["t"]
        self.q = d["q"][:, SDK_TO_MODEL]
        self.dq = d["dq"][:, SDK_TO_MODEL]
        self.tau = d["tau_est"][:, SDK_TO_MODEL]
        self.t0, self.c0, self.c1, self.s = t0, c0, c1, stretch

    def rec_time(self, ts):
        """sim time (0 = start of replay) -> (recording time, d rec / d sim)."""
        tr = self.t0 + ts
        if tr <= self.c0:
            return tr, 1.0
        if tr <= self.c0 + (self.c1 - self.c0) * self.s:
            return self.c0 + (tr - self.c0) / self.s, 1.0 / self.s
        return tr - (self.c1 - self.c0) * (self.s - 1.0), 1.0

    def __call__(self, ts):
        tr, rate = self.rec_time(ts)
        i = int(np.clip(np.searchsorted(self.t, tr), 1, len(self.t) - 1))
        a = (tr - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        lerp = lambda x: (1 - a) * x[i - 1] + a * x[i]
        return lerp(self.q), lerp(self.dq) * rate, lerp(self.tau), tr, rate

    def duration(self, t_end):
        return t_end - self.t0 + (self.c1 - self.c0) * (self.s - 1.0)


def run(args):
    d = np.load(args.rec)
    meta = json.load(open(args.rec.replace(".npz", ".json")))
    t_req = next(e["t"] for e in meta["events"]
                 if e["event"] == "request" and e["api_id"] == 1031)
    c0, c1 = find_phases(d, t_req)
    t0 = t_req - 0.5
    ref = Reference(d, t0, c0, c1, args.stretch)

    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = args.dt
    if args.torque_scale != 1.0:
        m.actuator_ctrlrange[:] *= args.torque_scale
    lim = m.actuator_ctrlrange[:, 1].copy()
    data = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # start in the recorded stance, then let the body settle under PD
    q_init = ref(0.0)[0]
    data.qpos[:3] = [0, 0, 0.40]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = q_init
    mujoco.mj_forward(m, data)
    for _ in range(int(1.0 / args.dt)):
        data.ctrl[:] = np.clip(60 * (q_init - data.qpos[7:]) - 5 * data.qvel[6:], -lim, lim)
        mujoco.mj_step(m, data)

    renderer = cam = frames = None
    if args.video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.6, -15, 120
        frames = []

    T = ref.duration(min(d["t"][-1], t_req + args.post))
    n = int(T / args.dt)
    log = {"z": [], "x": [], "tilt": [], "air": [], "sat": [], "tr": []}
    x_start = data.qpos[0]
    for k in range(n):
        ts = k * args.dt
        q_r, dq_r, tau_r, tr, rate = ref(ts)
        ff = args.ff * tau_r
        if rate != 1.0 and args.crouch_ff == "none":
            ff = 0.0 * tau_r
        tau = args.kp * (q_r - data.qpos[7:]) + args.kd * (dq_r - data.qvel[6:]) + ff
        sat = np.abs(tau) > lim
        data.ctrl[:] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, data)

        if k % int(0.002 / args.dt) == 0:
            touching = any(floor in (c.geom1, c.geom2) for c in data.contact[:data.ncon])
            w, x, y, z = data.qpos[3:7]
            up_z = 1 - 2 * (x * x + y * y)          # body z-axis · world z
            log["z"].append(data.qpos[2]); log["x"].append(data.qpos[0] - x_start)
            log["tilt"].append(np.degrees(np.arccos(np.clip(up_z, -1, 1))))
            log["air"].append(not touching); log["sat"].append(sat.any()); log["tr"].append(tr)
        if renderer is not None and k % int(1 / (30 * args.dt)) == 0:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    air = np.array(log["air"]); tr = np.array(log["tr"])
    tilt = np.array(log["tilt"]); z = np.array(log["z"])
    # longest continuous airborne stretch
    best = cur = 0
    for a in air:
        cur = cur + 1 if a else 0
        best = max(best, cur)
    after = tr > c1
    res = {
        "stretch": args.stretch, "kp": args.kp, "kd": args.kd, "ff": args.ff,
        "torque_scale": args.torque_scale,
        "crouch_rec_s": round(c1 - c0, 3), "crouch_sim_s": round((c1 - c0) * args.stretch, 3),
        "z_min": round(float(z.min()), 3), "z_max_after_crouch": round(float(z[after].max()), 3),
        "airborne_s": round(best * 0.002, 3),
        "forward_m": round(float(log["x"][-1]), 3),
        "tilt_max_deg": round(float(tilt.max()), 1), "tilt_final_deg": round(float(tilt[-1]), 1),
        "saturated_frac": round(float(np.mean(log["sat"])), 3),
        "upright_at_end": bool(tilt[-1] < 30 and z[-1] > 0.15),
    }
    print(json.dumps(res, ensure_ascii=False))
    if frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=30)
        print(f"video -> {args.video} ({len(frames)} frames)")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("--stretch", type=float, default=1.0, help="crouch slow-down factor")
    p.add_argument("--kp", type=float, default=40.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--ff", type=float, default=1.0, help="scale on recorded tau_est")
    p.add_argument("--crouch-ff", choices=["recorded", "none"], default="recorded",
                   help="feed-forward during the stretched crouch")
    p.add_argument("--torque-scale", type=float, default=1.0,
                   help="scale menagerie actuator limits (23.7 / 45.4 Nm)")
    p.add_argument("--post", type=float, default=3.0, help="replay until t_req + post [s]")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--video")
    run(p.parse_args())


if __name__ == "__main__":
    main()
