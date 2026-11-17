#!/usr/bin/env python3
"""Hand the Go2's motors over from the built-in sport controller to low level.

The moment this succeeds the legs go slack, so the robot must be suspended.
Method and api_id taken from the go2_control node already validated on this robot.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request, Response

API_ID_RELEASE_MODE = 1003


class Release(Node):
    def __init__(self):
        super().__init__("release_sport_mode")
        self.pub = self.create_publisher(Request, "/api/motion_switcher/request", 10)
        self.create_subscription(Response, "/api/motion_switcher/response", self._on_res, 10)
        self.req_id = time.time_ns()
        self.done = False
        self.tries = 0
        self.create_timer(1.0, self._send)

    def _send(self):
        self.tries += 1
        if self.tries > 8:
            self.get_logger().error("no response after 8 attempts")
            raise SystemExit(1)
        req = Request()
        req.header.identity.api_id = API_ID_RELEASE_MODE
        req.header.identity.id = self.req_id
        self.pub.publish(req)
        self.get_logger().info(f"sent ReleaseMode ({self.tries})")

    def _on_res(self, msg: Response):
        if msg.header.identity.api_id != API_ID_RELEASE_MODE:
            return
        self.get_logger().info(f"response: {msg.data}")
        self.done = True
        raise SystemExit(0)


def main():
    rclpy.init()
    node = Release()
    code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = int(e.code or 0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
