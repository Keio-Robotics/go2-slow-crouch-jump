#!/usr/bin/env python3
"""Full sequence in MuJoCo with the exact JumpPlan the real node uses:
lying folded on the floor -> stand up -> trigger -> slow crouch + recorded jump
-> recover -> sit.  Targets update at --rate Hz with --delay ms of latency; the
motor PD runs every physics step like the real motor driver."""

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from jump_plan import SIT_POS, JumpPlan

# MuJoCo Menagerie Go2 scene; clone https://github.com/google-deepmind/mujoco_menagerie
# next to this file, or point GO2_SCENE at scene.xml.
SCENE = os.environ.get("GO2_SCENE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_go2", "scene.xml"))
SDK_TO_MODEL = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
MODEL_TO_SDK = np.argsort(SDK_TO_MODEL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jump", default="data/jump01_replay.npz")
    p.add_argument("--stretch", type=float, default=10.0)
    p.add_argument("--ff-scale", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=200.0)
    p.add_argument("--delay", type=float, default=0.0, help="command latency [ms]")
    p.add_argument("--torque-scale", type=float, default=1.0)
    p.add_argument("--trigger-at", type=float, default=6.0)
    p.add_argument("--jumps", type=int, default=1, help="hold the trigger for this many jumps")
    p.add_argument("--video")
    a = p.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = 0.001
    m.actuator_ctrlrange[:] *= a.torque_scale
    lim = m.actuator_ctrlrange[:, 1].copy()
    d = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[:3] = [0, 0, 0.15]; d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = SIT_POS[SDK_TO_MODEL]
    for _ in range(1500):  # damped, settle lying on the floor
        d.ctrl[:] = np.clip(-5 * d.qvel[6:], -lim, lim); mujoco.mj_step(m, d)

    plan = JumpPlan(a.jump, d.qpos[7:][MODEL_TO_SDK], stretch=a.stretch, ff_scale=a.ff_scale)
    cycle = 0.5 + plan.jump_duration() + plan.RECOVER + plan.MIN_STAND
    t_sit = a.trigger_at + a.jumps * cycle + 0.5
    T = t_sit + plan.SIT + 1.0
    every = int(round(1.0 / (a.rate * m.opt.timestep)))
    lag = int(round(a.delay / 1000 / m.opt.timestep))
    queue = []
    frames, r, cam = [], None, None
    if a.video:
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 1.8, -12, 110
    st = {"air": 0, "air_best": 0, "tilt_max": 0.0, "v_max_nonjump": 0.0, "v_max": 0.0,
          "sat_jump": 0, "n_jump": 0, "err_max": {}, "x_trig": None}
    states = []
    cmd = None
    for k in range(int(T / m.opt.timestep)):
        t = k * m.opt.timestep
        if k % every == 0:
            s, q, dq, kp, kd, ff = plan.step(t, trigger=a.trigger_at <= t < a.trigger_at + (a.jumps - 1) * cycle + 0.3, sit=t >= t_sit)
            queue.append((k + lag, s, q[SDK_TO_MODEL], dq[SDK_TO_MODEL], kp, kd, ff[SDK_TO_MODEL]))
        while queue and queue[0][0] <= k:
            cmd = queue.pop(0)[1:]
        if cmd is None:
            mujoco.mj_step(m, d); continue
        s, q, dq, kp, kd, ff = cmd
        if s == "done":
            kp, kd, ff = 0.0, 5.0, np.zeros(12)
        tau = kp * (q - d.qpos[7:]) + kd * (dq - d.qvel[6:]) + ff
        d.ctrl[:] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if not states or states[-1] != s:
            states.append(s)
            if s == "prep":
                st["x_trig"] = float(d.qpos[0])
        w, x, y, _ = d.qpos[3:7]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        st["tilt_max"] = max(st["tilt_max"], tilt)
        v = float(np.abs(d.qvel[6:]).max())
        st["v_max"] = max(st["v_max"], v)
        if s != "jump":
            st["v_max_nonjump"] = max(st["v_max_nonjump"], v)
        else:
            st["n_jump"] += 1; st["sat_jump"] += int((np.abs(tau) > lim).any())
        if s not in ("hold",):
            st["err_max"][s] = max(st["err_max"].get(s, 0.0), float(np.abs(q - d.qpos[7:]).max()))
        touching = any(floor in (c.geom1, c.geom2) and (c.geom1 != floor or c.geom2 != floor)
                       and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1]) != "world"
                       for c in d.contact[:d.ncon])
        st["air"] = st["air"] + 1 if not touching else 0
        st["air_best"] = max(st["air_best"], st["air"])
        if s == "recover" and "x_land" not in st:
            st["x_land"] = float(d.qpos[0]); st["tilt_at_recover"] = round(float(tilt), 1)
        if s == "sit" and "z_before_sit" not in st:
            st["z_before_sit"] = round(float(d.qpos[2]), 3); st["tilt_before_sit"] = round(float(tilt), 1)
        if r is not None and k % 33 == 0:
            cam.lookat[:] = d.qpos[:3]; r.update_scene(d, cam); frames.append(r.render())
    out = {"jumps_done": plan.jumps, "stretch": a.stretch, "ff_scale": a.ff_scale, "rate": a.rate, "delay_ms": a.delay,
           "torque_scale": a.torque_scale, "states": states,
           "airborne_s": st["air_best"] * m.opt.timestep,
           "forward_m": round(st.get("x_land", np.nan) - (st["x_trig"] or 0), 3),
           "tilt_max_deg": round(st["tilt_max"], 1), "tilt_at_recover": st.get("tilt_at_recover"),
           "z_before_sit": st.get("z_before_sit"), "tilt_before_sit": st.get("tilt_before_sit"),
           "jvel_max_nonjump": round(st["v_max_nonjump"], 1), "jvel_max": round(st["v_max"], 1),
           "sat_frac_jump": round(st["sat_jump"] / max(st["n_jump"], 1), 3),
           "err_max": {k: round(v, 2) for k, v in st["err_max"].items()},
           "final_z": round(float(d.qpos[2]), 3)}
    print(json.dumps(out))
    if frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=30)


if __name__ == "__main__":
    main()
