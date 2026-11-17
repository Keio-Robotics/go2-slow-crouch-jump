#!/usr/bin/env python3
"""Suspended test: replay the recorded FrontJump crouch slowly, then come back up.

Runs on the Jetson payload (unitree_ros2 sourced). The robot must be HANGING and
sport mode must already be released (release_sport_mode.py).

    DRY RUN (default)  reads /lowstate, steps the plan, prints targets, publishes nothing
    --arm              publishes /lowcmd

Only the crouch segment exists in the input file (crouch_plan.extract_crouch drops
thrust, flight and landing), so this node has nothing that could jump.

Safety, mirroring go2_deploy_node.py (damping = kp 0, kd 5, latched):
  remote L2+B, Ctrl-C / SIGTERM, stale telemetry, tilt, joint speed, tracking error.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from crouch_plan import CrouchPlan
from go2_crc import lowcmd_crc

NUM_JOINTS = 12
RATE_HZ = 100
DAMP_KD = 5.0

MAX_TILT_RAD = 0.6
MAX_JOINT_VEL = 8.0       # the plan never asks for more than ~2 rad/s at stretch 10
MAX_TRACK_ERR = 0.6       # rad, checked once gains are fully up
MAX_STATE_AGE = 0.15

BTN_L2, BTN_B = 5, 9
WHEEL_INDICES = (12, 13, 14, 15)
WHEEL_ACTIVITY_TOL = 0.05


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


class CrouchNode(Node):
    def __init__(self, args):
        super().__init__("go2_crouch_replay")
        self.args = args
        self.q = np.zeros(NUM_JOINTS)
        self.dq = np.zeros(NUM_JOINTS)
        self.grav = np.array([0.0, 0.0, -1.0])
        self.buttons = 0
        self.state_time = 0.0
        self.have_state = False
        self.model_checked = False
        self.damped = False
        self.plan = None
        self.t0 = 0.0
        self.n = 0
        self.max_err = 0.0
        self.last_phase = ""

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_state, qos)
        self.pub = self.create_publisher(LowCmd, "/lowcmd", 1) if args.arm else None
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self.create_timer(1.0 / RATE_HZ, self._step)
        mode = "ARMED — WILL PUBLISH /lowcmd" if args.arm else "DRY RUN — publishes nothing"
        self.get_logger().warning(f"*** {mode} *** stretch={args.stretch} kp={args.kp} kd={args.kd}")

    # ── telemetry ──
    def _on_state(self, m: LowState):
        for i in range(NUM_JOINTS):
            self.q[i] = m.motor_state[i].q
            self.dq[i] = m.motor_state[i].dq
        self.grav = quat_rotate_inverse(np.array(m.imu_state.quaternion), np.array([0.0, 0.0, -1.0]))
        if not self.model_checked:
            self._identify(m)
        try:
            raw = bytes(bytearray(m.wireless_remote))
            self.buttons = int(raw[2]) | (int(raw[3]) << 8)
        except Exception:
            self.buttons = 0
        if (self.buttons >> BTN_L2) & 1 and (self.buttons >> BTN_B) & 1:
            self._trip("remote L2+B")
        self.state_time = time.monotonic()
        self.have_state = True

    def _identify(self, m):
        self.model_checked = True
        host = socket.gethostname()
        exp = self.args.expected_hostname
        if exp and host != exp:
            raise SystemExit(f"hostname {host!r} != {exp!r}; refusing")
        if any(abs(m.motor_state[i].dq) > WHEEL_ACTIVITY_TOL for i in WHEEL_INDICES):
            raise SystemExit("wheel motors turning: looks like a Go2W, refusing")
        self.get_logger().info(f"robot identity OK (host={host})")

    # ── safety ──
    def _trip(self, reason):
        if not self.damped:
            self.damped = True
            self.get_logger().error(f"DAMPING — {reason}")

    def _on_signal(self, *_):
        self._trip("operator signal")
        self._publish_damping()
        rclpy.shutdown()
        sys.exit(0)

    def _check(self, phase, q_t):
        age = time.monotonic() - self.state_time
        if age > MAX_STATE_AGE:
            self._trip(f"telemetry stale {age * 1000:.0f} ms")
        tilt = float(np.linalg.norm(self.grav[:2]))
        if tilt > np.sin(MAX_TILT_RAD):
            self._trip(f"tilt {np.degrees(np.arcsin(min(tilt, 1.0))):.0f} deg")
        v = float(np.abs(self.dq).max())
        if v > MAX_JOINT_VEL:
            self._trip(f"joint speed {v:.1f} rad/s")
        err = float(np.abs(q_t - self.q).max())
        self.max_err = max(self.max_err, err) if phase != "hold" else self.max_err
        if phase != "hold" and err > MAX_TRACK_ERR:
            if self.args.arm:
                self._trip(f"tracking error {err:.2f} rad in phase {phase}")
            elif not getattr(self, "_warned_err", False):
                self._warned_err = True  # dry run: the robot is not following, expected
                self.get_logger().warning(f"[DRY] would trip: tracking error {err:.2f} rad in {phase}")

    # ── loop ──
    def _step(self):
        if not self.have_state:
            return
        if self.plan is None:
            self.plan = CrouchPlan(self.args.crouch, self.q, stretch=self.args.stretch, kp=self.args.kp)
            self.t0 = time.monotonic()
            self.get_logger().info(
                "plan: " + ", ".join(f"{n} {a:.1f}-{b:.1f}s" for n, a, b in self.plan.phases))
        t = time.monotonic() - self.t0
        phase, q_t, dq_t, kp = self.plan(t)
        if self.damped and phase != "done":
            self._publish_damping()
            return
        if phase == "done":
            if not self.damped:
                self.get_logger().info(f"plan finished; max tracking error {self.max_err:.3f} rad")
            self._trip("planned end (robot must be suspended)")
            self._publish_damping()
            if t > self.plan.T + 1.0:
                raise SystemExit(0)
            return
        self._check(phase, q_t)
        if self.damped:
            self._publish_damping()
            return
        if phase != self.last_phase:
            self.get_logger().info(f"phase: {phase} (t={t:.2f}s)")
            self.last_phase = phase
        self.n += 1
        if self.n % (RATE_HZ // 2) == 0:
            err = q_t - self.q
            self.get_logger().info(
                f"[{'ARM' if self.args.arm else 'DRY'}] {phase} t={t:.2f} kp={kp:.1f} "
                f"max|err|={np.abs(err).max():.3f} calf tgt={np.array2string(q_t[[2, 5, 8, 11]], precision=2)} "
                f"now={np.array2string(self.q[[2, 5, 8, 11]], precision=2)}")
        self._publish(q_t, dq_t, kp, self.args.kd)

    # ── outgoing ──
    def _fill(self, cmd, q, dq, kp, kd):
        cmd.head = [0xFE, 0xEF]
        cmd.level_flag = 0xFF
        for s in range(NUM_JOINTS):
            mc = cmd.motor_cmd[s]
            mc.mode = 0x01
            mc.q = float(q[s]) if q is not None else 0.0
            mc.dq = float(dq[s]) if dq is not None else 0.0
            mc.tau = 0.0
            mc.kp = float(kp)
            mc.kd = float(kd)

    def _publish(self, q, dq, kp, kd):
        if self.pub is None:
            return
        cmd = LowCmd()
        self._fill(cmd, q, dq, kp, kd)
        cmd.crc = lowcmd_crc(cmd)
        self.pub.publish(cmd)

    def _publish_damping(self):
        if self.pub is None:
            return
        cmd = LowCmd()
        self._fill(cmd, None, None, 0.0, DAMP_KD)
        cmd.crc = lowcmd_crc(cmd)
        self.pub.publish(cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crouch", default="data/crouch01.npz", help="output of crouch_plan.extract_crouch")
    p.add_argument("--arm", action="store_true")
    p.add_argument("--stretch", type=float, default=10.0)
    p.add_argument("--kp", type=float, default=40.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--expected-hostname", default="", help="refuse to run on any other host (optional)")
    args = p.parse_args()
    if args.stretch < 5.0:
        raise SystemExit("stretch < 5 is not allowed for the suspended crouch test")
    rclpy.init()
    node = CrouchNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish_damping()
        except Exception as exc:
            print(f"[warn] could not publish final damping: {exc}", file=sys.stderr)
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
