#!/usr/bin/env python3
"""Real Go2: stand up from damping, wait, and on the operator's remote signal do a
slow crouch followed by the recorded FrontJump thrust/flight/landing (open loop).

Runs on the Jetson payload (unitree_ros2 sourced). Sport mode must already be
released (release_sport_mode.py); the node checks this and refuses otherwise.

    DRY RUN (default)  reads /lowstate, steps the plan, prints targets, publishes nothing
    --arm              publishes /lowcmd

Remote (Unitree handheld):
    R1 + A   (press both, robot standing)  -> jump; release and press again to repeat
             (at least 2 s of standing between jumps)
    L1       (robot standing)              -> sit down, then damping, exit
    L2 + B   (any time)                    -> damping immediately (latched)
Ctrl-C / SIGTERM also damp. Damping = kp 0, kd 5 on every joint.

Watchdogs (-> damping): stale telemetry, tilt > 34 deg, joint speed, tracking error,
motor temperature >= 75 C.
Speed / error limits are looser during the jump itself, where large values are normal
and cutting power mid-air would be worse than finishing the landing.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_api.msg import Request, Response
from unitree_go.msg import LowCmd, LowState

from go2_crc import lowcmd_crc
from jump_plan import JumpPlan

NUM_JOINTS = 12
RATE_HZ = 200
DAMP_KD = 5.0
MAX_STATE_AGE = 0.15
MAX_TILT_RAD = 0.6
MAX_MOTOR_TEMP = 75       # deg C; long slow crouches hold load for minutes
WARN_MOTOR_TEMP = 65
# (joint speed rad/s, tracking error rad) per plan state
LIMITS = {
    "hold": (8.0, None), "stand_up": (8.0, 0.8), "standing": (8.0, 0.6),
    "prep": (8.0, 0.6), "crouch": (8.0, 0.6), "jump": (30.0, 1.2),
    "recover": (15.0, 0.8), "sit": (8.0, 0.8), "done": (30.0, None),
}

BTN_R1, BTN_L1, BTN_L2, BTN_A, BTN_B = 0, 1, 5, 8, 9
WHEEL_INDICES = (12, 13, 14, 15)
WHEEL_ACTIVITY_TOL = 0.05
API_SWITCHER_CHECKMODE = 1001


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v * (2.0 * w * w - 1.0) - np.cross(qv, v) * w * 2.0 + qv * np.dot(qv, v) * 2.0


class JumpNode(Node):
    def __init__(self, args):
        super().__init__("go2_jump_replay")
        self.args = args
        self.q = np.zeros(NUM_JOINTS); self.dq = np.zeros(NUM_JOINTS)
        self.tau_est = np.zeros(NUM_JOINTS); self.foot = np.zeros(4)
        self.temp = np.zeros(NUM_JOINTS)
        self.temp_warned = False
        self.grav = np.array([0.0, 0.0, -1.0])
        self.buttons = 0
        self.state_time = 0.0
        self.have_state = False
        self.model_checked = False
        self.mode_ok = None          # None = unknown, True = sport released
        self.damped = False
        self.plan = None
        self.t0 = 0.0
        self.trigger_armed = False   # trigger needs R1+A released once before it counts
        self.last_state = ""
        self.n = 0
        self.log = []

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_state, qos)
        self.create_subscription(Response, "/api/motion_switcher/response", self._on_mode, 10)
        self.mode_pub = self.create_publisher(Request, "/api/motion_switcher/request", 10)
        self.pub = self.create_publisher(LowCmd, "/lowcmd", 1) if args.arm else None
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self.create_timer(1.0 / RATE_HZ, self._step)
        self.create_timer(0.5, self._ask_mode)
        mode = "ARMED — WILL PUBLISH /lowcmd" if args.arm else "DRY RUN — publishes nothing"
        self.get_logger().warning(f"*** {mode} *** stretch={args.stretch} ff_scale={args.ff_scale}")

    # ── sport mode check ──
    def _ask_mode(self):
        if self.mode_ok is not None:
            return
        r = Request()
        r.header.identity.id = time.time_ns()
        r.header.identity.api_id = API_SWITCHER_CHECKMODE
        self.mode_pub.publish(r)

    def _on_mode(self, m: Response):
        if m.header.identity.api_id != API_SWITCHER_CHECKMODE or self.mode_ok is not None:
            return
        try:
            name = json.loads(m.data).get("name", "?")
        except Exception:
            name = "?"
        self.mode_ok = (name == "")
        if self.mode_ok:
            self.get_logger().info("sport mode is released")
        else:
            self.get_logger().error(f"sport mode is ACTIVE ({name!r}); run release_sport_mode.py first")
            if self.args.arm:
                raise SystemExit(2)

    # ── telemetry ──
    def _on_state(self, m: LowState):
        for i in range(NUM_JOINTS):
            ms = m.motor_state[i]
            self.q[i], self.dq[i], self.tau_est[i] = ms.q, ms.dq, ms.tau_est
            self.temp[i] = ms.temperature
        self.foot[:] = m.foot_force[:4]
        self.grav = quat_rotate_inverse(np.array(m.imu_state.quaternion), np.array([0.0, 0.0, -1.0]))
        if not self.model_checked:
            self._identify(m)
        try:
            raw = bytes(bytearray(m.wireless_remote))
            self.buttons = int(raw[2]) | (int(raw[3]) << 8)
        except Exception:
            self.buttons = 0
        if self._btn(BTN_L2) and self._btn(BTN_B):
            self._trip("remote L2+B")
        self.state_time = time.monotonic()
        self.have_state = True

    def _btn(self, b):
        return bool((self.buttons >> b) & 1)

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
        raise SystemExit(0)

    def _check(self, state, q_t):
        age = time.monotonic() - self.state_time
        if age > MAX_STATE_AGE:
            self._trip(f"telemetry stale {age * 1000:.0f} ms")
        tilt = float(np.linalg.norm(self.grav[:2]))
        if tilt > np.sin(MAX_TILT_RAD):
            self._trip(f"tilt {np.degrees(np.arcsin(min(tilt, 1.0))):.0f} deg in {state}")
        tmax = float(self.temp.max())
        if tmax >= MAX_MOTOR_TEMP:
            self._trip(f"motor temperature {tmax:.0f} C (joint {int(self.temp.argmax())}) in {state}")
        elif tmax >= WARN_MOTOR_TEMP and not self.temp_warned:
            self.temp_warned = True
            self.get_logger().warning(f"motor temperature {tmax:.0f} C (trip at {MAX_MOTOR_TEMP})")
        vlim, elim = LIMITS.get(state, (8.0, 0.6))
        v = float(np.abs(self.dq).max())
        if v > vlim:
            self._trip(f"joint speed {v:.1f} rad/s in {state}")
        if elim is not None:
            err = float(np.abs(q_t - self.q).max())
            if err > elim:
                if self.args.arm:
                    self._trip(f"tracking error {err:.2f} rad in {state}")
                elif not getattr(self, "_warned", False):
                    self._warned = True
                    self.get_logger().warning(f"[DRY] would trip: tracking error {err:.2f} rad in {state}")

    # ── loop ──
    def _step(self):
        if not self.have_state or self.mode_ok is None:
            return
        now = time.monotonic()
        if self.plan is None:
            self.plan = JumpPlan(self.args.traj, self.q, stretch=self.args.stretch, ff_scale=self.args.ff_scale)
            if self.args.crouch_secs:
                self.plan.s = self.args.crouch_secs / self.plan.tc
            self.get_logger().info(f"crouch {self.plan.tc * self.plan.s:.1f}s (stretch {self.plan.s:.0f}x)")
            self.t0 = now
            self.get_logger().info(f"plan loaded; jump section {self.plan.jump_duration():.2f}s")
        t = now - self.t0
        if self.damped:
            self._publish_damping()
            if not hasattr(self, "_damp_t"):
                self._damp_t = now
            elif now - self._damp_t > 1.0:
                raise SystemExit(0)
            return

        both = self._btn(BTN_R1) and self._btn(BTN_A)
        if self.plan.state == "standing" and not both:
            self.trigger_armed = True
        trigger = self.trigger_armed and both
        sit = self._btn(BTN_L1) or (self.plan.state == "standing" and t - self.plan.t_state > self.args.max_wait)
        if self.args.dry_trigger_at and t >= self.args.dry_trigger_at and not self.args.arm:
            trigger = True
        state, q_t, dq_t, kp, kd, ff = self.plan.step(t, trigger=trigger, sit=sit)
        if state != self.last_state:
            if state == "prep":
                self.trigger_armed = False   # must release R1+A before the next jump
                self.get_logger().warning(f"JUMP #{self.plan.jumps}")
            self.get_logger().warning(f"state: {state} (t={t:.2f}s)") if state in ("prep", "jump") \
                else self.get_logger().info(f"state: {state} (t={t:.2f}s)")
            if state == "standing" and self.last_state == "stand_up":
                self.get_logger().warning("STANDING — press R1+A to jump, L1 to sit, L2+B to damp")
            self.last_state = state
        if state == "done":
            self.get_logger().info("sat down; damping")
            self._trip("planned end")
            self._publish_damping()
            return
        self._check(state, q_t)
        if self.damped:
            self._publish_damping()
            return
        self.n += 1
        self.log.append([t, *self.q, *self.dq, *self.tau_est, *self.foot, *q_t, *ff, kp, kd,
                         ["hold", "stand_up", "standing", "prep", "crouch", "jump", "recover", "sit"].index(state)
                         if state in ("hold", "stand_up", "standing", "prep", "crouch", "jump", "recover", "sit") else -1])
        if self.n % RATE_HZ == 0 and state not in ("jump",):
            self.get_logger().info(
                f"[{'ARM' if self.args.arm else 'DRY'}] {state} t={t:.1f} kp={kp:.0f} kd={kd:.1f} "
                f"max|err|={np.abs(q_t - self.q).max():.3f} feet={np.array2string(self.foot, precision=0)} "
                f"Tmax={self.temp.max():.0f}C")
        self._publish(q_t, dq_t, kp, kd, ff)

    # ── outgoing ──
    def _fill(self, cmd, q, dq, kp, kd, ff):
        cmd.head = [0xFE, 0xEF]
        cmd.level_flag = 0xFF
        for s in range(NUM_JOINTS):
            mc = cmd.motor_cmd[s]
            mc.mode = 0x01
            mc.q = float(q[s]) if q is not None else 0.0
            mc.dq = float(dq[s]) if dq is not None else 0.0
            mc.tau = float(ff[s]) if ff is not None else 0.0
            mc.kp = float(kp)
            mc.kd = float(kd)

    def _publish(self, q, dq, kp, kd, ff):
        if self.pub is None:
            return
        cmd = LowCmd()
        self._fill(cmd, q, dq, kp, kd, ff)
        cmd.crc = lowcmd_crc(cmd)
        self.pub.publish(cmd)

    def _publish_damping(self):
        if self.pub is None:
            return
        cmd = LowCmd()
        self._fill(cmd, None, None, 0.0, DAMP_KD, None)
        cmd.crc = lowcmd_crc(cmd)
        self.pub.publish(cmd)

    def save(self):
        if self.log:
            np.save(self.args.log, np.asarray(self.log, dtype=np.float64))
            print(f"log -> {self.args.log} ({len(self.log)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default="data/jump01_replay.npz", help="output of jump_plan.extract_jump")
    p.add_argument("--arm", action="store_true")
    p.add_argument("--stretch", type=float, default=10.0)
    p.add_argument("--crouch-secs", type=float, default=0.0,
                   help="crouch duration in seconds (overrides --stretch), e.g. 600 for 10 min")
    p.add_argument("--ff-scale", type=float, default=1.0)
    p.add_argument("--expected-hostname", default="", help="refuse to run on any other host (optional)")
    p.add_argument("--max-wait", type=float, default=300.0, help="sit down after this long standing idle [s]")
    p.add_argument("--log", default=time.strftime("jump_replay_%Y%m%d_%H%M%S.npy"))
    p.add_argument("--dry-trigger-at", type=float, default=0.0,
                   help="dry run only: fake the trigger at this time to exercise the jump path")
    args = p.parse_args()
    if args.stretch < 1.0 or (args.crouch_secs and args.crouch_secs < 0.5):
        raise SystemExit("stretch must be >= 1")
    rclpy.init()
    node = JumpNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code not in (0, None):
            print(f"exit: {e.code}", file=sys.stderr)
    finally:
        try:
            if rclpy.ok():
                node._publish_damping()
        except Exception as exc:
            print(f"[warn] could not publish final damping: {exc}", file=sys.stderr)
        node.save()
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
