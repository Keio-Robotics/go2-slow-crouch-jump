#!/usr/bin/env python3
"""Print the 12 leg-motor temperatures (LowState.motor_state[i].temperature, deg C)."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowState

NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]


def main():
    rclpy.init()
    n = Node("motor_temps")
    got = []
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    n.create_subscription(LowState, "/lowstate", lambda m: got.append(m), qos)
    t0 = time.time()
    while not got and time.time() - t0 < 3:
        rclpy.spin_once(n, timeout_sec=0.1)
    if not got:
        raise SystemExit("no /lowstate received")
    m = got[-1]
    for i, name in enumerate(NAMES):
        print(f"{name:9s} {m.motor_state[i].temperature:4d} C")
    print(f"battery soc {m.bms_state.soc}%")


if __name__ == "__main__":
    main()
