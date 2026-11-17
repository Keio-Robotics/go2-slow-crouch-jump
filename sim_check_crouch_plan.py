#!/usr/bin/env python3
"""Run CrouchPlan (the exact targets go2_crouch_replay.py sends) in MuJoCo.

--suspended pins the base in the air like the real test; otherwise the robot
stands on the floor. Targets update at the node's 100 Hz; the motor PD
(kp*(q*-q) + kd*(dq*-dq)) runs every physics step, as on the real motor driver.
"""

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from crouch_plan import CrouchPlan

# MuJoCo Menagerie Go2 scene; clone https://github.com/google-deepmind/mujoco_menagerie
# next to this file, or point GO2_SCENE at scene.xml.
SCENE = os.environ.get("GO2_SCENE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_go2", "scene.xml"))
SDK_TO_MODEL = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
MODEL_TO_SDK = np.argsort(SDK_TO_MODEL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crouch", default="data/crouch01.npz")
    p.add_argument("--stretch", type=float, default=10.0)
    p.add_argument("--kp", type=float, default=40.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--suspended", action="store_true")
    p.add_argument("--video")
    a = p.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = 0.001
    lim = m.actuator_ctrlrange[:, 1].copy()
    d = mujoco.MjData(m)
    # a slack-ish starting pose different from the plan start, like a hanging robot
    q_start_sdk = np.array([0.0, 1.2, -2.4] * 4) if a.suspended else None
    d.qpos[:3] = [0, 0, 0.6 if a.suspended else 0.40]
    d.qpos[3:7] = [1, 0, 0, 0]
    plan_probe = np.load(a.crouch)["q"][0]
    d.qpos[7:] = (q_start_sdk if a.suspended else plan_probe)[SDK_TO_MODEL]
    mujoco.mj_forward(m, d)
    base0 = d.qpos[:7].copy()
    if not a.suspended:  # settle on the floor holding the start pose
        for _ in range(1000):
            d.ctrl[:] = np.clip(60 * (plan_probe[SDK_TO_MODEL] - d.qpos[7:]) - 5 * d.qvel[6:], -lim, lim)
            mujoco.mj_step(m, d)

    q_sdk = d.qpos[7:][MODEL_TO_SDK]
    plan = CrouchPlan(a.crouch, q_sdk, stretch=a.stretch, kp=a.kp)
    frames, r, cam = [], None, None
    if a.video:
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 1.6, -15, 120
    max_err, max_tau, max_v, z_min = 0.0, np.zeros(12), 0.0, 9.0
    sat = 0
    n = int((plan.T + 0.5) / m.opt.timestep)
    for k in range(n):
        t = k * m.opt.timestep
        if k % 10 == 0:  # 100 Hz target update
            phase, q_t, dq_t, kp = plan(t)
            q_t, dq_t = q_t[SDK_TO_MODEL], dq_t[SDK_TO_MODEL]
        tau = kp * (q_t - d.qpos[7:]) + a.kd * (dq_t - d.qvel[6:])
        sat += int((np.abs(tau) > lim).any())
        d.ctrl[:] = np.clip(tau, -lim, lim)
        if a.suspended:
            d.qpos[:7] = base0; d.qvel[:6] = 0
        mujoco.mj_step(m, d)
        if phase not in ("hold", "done"):
            max_err = max(max_err, float(np.abs(q_t - d.qpos[7:]).max()))
        max_tau = np.maximum(max_tau, np.abs(d.ctrl))
        max_v = max(max_v, float(np.abs(d.qvel[6:]).max()))
        z_min = min(z_min, float(d.qpos[2]))
        if r is not None and k % 33 == 0:
            cam.lookat[:] = d.qpos[:3]; r.update_scene(d, cam); frames.append(r.render())
    res = {"suspended": a.suspended, "stretch": a.stretch, "kp": a.kp, "kd": a.kd,
           "plan_s": round(plan.T, 2), "phases": [(nm, round(s, 2), round(e, 2)) for nm, s, e in plan.phases],
           "max_track_err_rad": round(max_err, 3), "max_joint_vel": round(max_v, 2),
           "max_tau_hip_thigh_calf": [round(float(max_tau[j::3].max()), 1) for j in range(3)],
           "saturated_steps": sat, "z_min": round(z_min, 3)}
    print(json.dumps(res))
    if frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=30)


if __name__ == "__main__":
    main()
