#!/usr/bin/env python3
"""Ask the built-in sport controller to lie the Go2 down (StandDown, api_id 1005).

Run this BEFORE release_sport_mode.py when the robot is standing on the floor:
releasing sport mode while standing makes the legs go slack and the robot drops.
"""

import time

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request, Response

API_STANDDOWN = 1005


def main():
    rclpy.init()
    n = Node("standdown_once")
    got = []
    n.create_subscription(
        Response, "/api/sport/response",
        lambda m: got.append(m.header.status.code) if m.header.identity.api_id == API_STANDDOWN else None, 10)
    pub = n.create_publisher(Request, "/api/sport/request", 10)
    t0, sent = time.time(), 0
    while time.time() - t0 < 6 and not got:
        if time.time() - t0 > 0.5 * sent and sent < 3:
            r = Request()
            r.header.identity.id = time.time_ns()
            r.header.identity.api_id = API_STANDDOWN
            pub.publish(r)
            sent += 1
        rclpy.spin_once(n, timeout_sec=0.05)
    print(f"StandDown sent {sent}x, response code: {got[0] if got else 'none'}")


if __name__ == "__main__":
    main()
