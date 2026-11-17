#!/usr/bin/env python3
"""Record LowState (and SportModeState) on a real Go2 while the built-in
FrontJump runs, so the joint trajectory can later be time-warped and replayed.

Runs on the Jetson payload with the unitree_ros2 environment sourced.
Never publishes /lowcmd. The only command it can send is a sport-mode request.

    (default)   record only for --duration seconds, send nothing
    --check-mode  ask the motion switcher which mode is active, then exit
    --jump      after --pre seconds send BalanceStand, then FrontJump,
                keep recording for --post seconds after the jump request

Output: <out>.npz with arrays (see save()) and <out>.json with events.
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
from unitree_go.msg import LowState, SportModeState

WHEEL_INDICES = (12, 13, 14, 15)
WHEEL_ACTIVITY_TOL = 0.5

API_BALANCESTAND = 1002
API_FRONTJUMP = 1031
API_SWITCHER_CHECKMODE = 1001


class Recorder(Node):
    def __init__(self, args):
        super().__init__("go2_record_jump")
        self.args = args
        self.t0 = time.monotonic()
        self.low = []    # rows: t, tick, q12, dq12, tau12, foot_force4, quat4, gyro3, acc3
        self.sport = []  # rows: t, mode, progress, body_height, pos3, vel3, rpy3
        self.events = []
        self.model_checked = False
        self.req_id = time.time_ns()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(LowState, "/lowstate", self._on_low, qos)
        self.create_subscription(SportModeState, "/sportmodestate", self._on_sport, qos)
        self.create_subscription(Response, "/api/sport/response", self._on_resp, 10)
        self.create_subscription(Response, "/api/motion_switcher/response", self._on_resp, 10)
        self.sport_pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.switch_pub = self.create_publisher(Request, "/api/motion_switcher/request", 10)

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self.stage = 0
        self.jump_t = None
        self.create_timer(0.02, self._tick)
        mode = "JUMP — will send BalanceStand + FrontJump" if args.jump else "RECORD ONLY — sends nothing"
        if args.check_mode:
            mode = "CHECK MODE — query motion switcher only"
        self.get_logger().warning(f"*** {mode} ***")

    def now(self) -> float:
        return time.monotonic() - self.t0

    def event(self, name, **kw):
        e = {"t": round(self.now(), 4), "event": name, **kw}
        self.events.append(e)
        self.get_logger().info(json.dumps(e, ensure_ascii=False))

    # ── callbacks ──
    def _on_low(self, m: LowState):
        if not self.model_checked:
            self._identify(m)
        ms = m.motor_state
        row = [self.now(), float(m.tick)]
        row += [ms[i].q for i in range(12)]
        row += [ms[i].dq for i in range(12)]
        row += [ms[i].tau_est for i in range(12)]
        row += list(m.foot_force)
        row += list(m.imu_state.quaternion) + list(m.imu_state.gyroscope) + list(m.imu_state.accelerometer)
        self.low.append(row)

    def _on_sport(self, m: SportModeState):
        self.sport.append([self.now(), float(m.mode), float(m.progress), m.body_height,
                           *m.position, *m.velocity, *m.imu_state.rpy])

    def _on_resp(self, m: Response):
        self.event("response", api_id=m.header.identity.api_id,
                   code=m.header.status.code, data=m.data)
        if self.args.check_mode and m.header.identity.api_id == API_SWITCHER_CHECKMODE:
            raise SystemExit(0)

    def _identify(self, m: LowState):
        self.model_checked = True
        host = socket.gethostname()
        exp = self.args.expected_hostname
        if exp and host != exp and not self.args.skip_model_check:
            raise SystemExit(f"hostname {host!r} != {exp!r}; refusing (Go2W shares the IP)")
        if any(abs(m.motor_state[i].dq) > WHEEL_ACTIVITY_TOL for i in WHEEL_INDICES) \
                and not self.args.skip_model_check:
            raise SystemExit("wheel motors turning: looks like a Go2W, refusing")
        self.event("identity_ok", host=host)

    def _send(self, pub, api_id, param=""):
        r = Request()
        r.header.identity.id = self.req_id
        self.req_id += 1
        r.header.identity.api_id = api_id
        r.parameter = param
        pub.publish(r)
        self.event("request", api_id=api_id, parameter=param)

    def _tick(self):
        t = self.now()
        a = self.args
        if a.check_mode:
            if self.stage == 0 and t > 0.5:
                self._send(self.switch_pub, API_SWITCHER_CHECKMODE)
                self.stage = 1
            if t > 5.0:
                raise SystemExit("no motion_switcher response in 5 s")
            return
        if not a.jump:
            if t > a.duration:
                raise SystemExit(0)
            return
        # jump sequence
        if self.stage == 0 and t > 1.0:
            if len(self.low) < 100:
                raise SystemExit("no /lowstate after 1 s; aborting before any command")
            if not self.sport:
                raise SystemExit("no /sportmodestate: sport mode may be released; aborting")
            self._send(self.sport_pub, API_BALANCESTAND)
            self.stage = 1
        elif self.stage == 1 and t > a.pre:
            self._send(self.sport_pub, API_FRONTJUMP, json.dumps({"data": True}))
            self.jump_t = t
            self.stage = 2
        elif self.stage == 2 and t > self.jump_t + a.post:
            raise SystemExit(0)

    def _on_signal(self, *_):
        raise SystemExit(130)

    def save(self):
        low = np.asarray(self.low, dtype=np.float64)
        sport = np.asarray(self.sport, dtype=np.float64)
        out = self.args.out
        if low.size:
            np.savez(out + ".npz",
                     t=low[:, 0], tick=low[:, 1],
                     q=low[:, 2:14], dq=low[:, 14:26], tau_est=low[:, 26:38],
                     foot_force=low[:, 38:42], quat_wxyz=low[:, 42:46],
                     gyro=low[:, 46:49], acc=low[:, 49:52],
                     sport=sport if sport.size else np.zeros((0, 13)),
                     sport_cols=np.array(["t", "mode", "progress", "body_height",
                                          "px", "py", "pz", "vx", "vy", "vz",
                                          "roll", "pitch", "yaw"]))
        hz = (len(low) - 1) / (low[-1, 0] - low[0, 0]) if len(low) > 1 else 0.0
        meta = {"args": vars(self.args), "n_low": len(low), "lowstate_hz": round(hz, 1),
                "n_sport": len(sport), "events": self.events,
                "joint_order": "SDK: FR(0-2) FL(3-5) RR(6-8) RL(9-11), hip/thigh/calf"}
        with open(out + ".json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"saved {out}.npz/.json  lowstate={len(low)} ({hz:.0f} Hz) sport={len(sport)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=time.strftime("jump_%Y%m%d_%H%M%S"))
    p.add_argument("--duration", type=float, default=5.0, help="record-only length [s]")
    p.add_argument("--check-mode", action="store_true")
    p.add_argument("--jump", action="store_true", help="send BalanceStand + FrontJump")
    p.add_argument("--pre", type=float, default=3.0, help="time before FrontJump [s]")
    p.add_argument("--post", type=float, default=4.0, help="record after FrontJump [s]")
    p.add_argument("--skip-model-check", action="store_true")
    p.add_argument("--expected-hostname", default="", help="refuse to run on any other host (optional)")
    args = p.parse_args()

    rclpy.init()
    node = Recorder(args)
    code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int) and e.code:
            print(f"stopped: {e.code}", file=sys.stderr)
    finally:
        if not args.check_mode:
            node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
