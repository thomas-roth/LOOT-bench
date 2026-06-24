"""Print the leader's live joint positions so you can pick a new home pose.

Move the leader arm by hand to wherever you want the robot to rest, read the
printed values, then copy them into _HOME_JOINT_POS in
g3a_bench/robots/trlc_dk1/dk1_constants.py.

Usage:
    uv run --extra cpu python g3a_bench/scripts/live_bridge/print_leader_pose.py
    uv run --extra cpu python g3a_bench/scripts/live_bridge/print_leader_pose.py --leader-port /dev/ttyUSB0
"""

import argparse
import time

from g3a_bench.scripts.live_bridge.reader import LiveReader

_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def main():
    parser = argparse.ArgumentParser(description="Print live leader joint positions")
    parser.add_argument(
        "--leader-port",
        type=str,
        default=None,
        help="Serial port for leader (e.g. /dev/ttyUSB0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=2.0,
        help="Print rate (Hz). Kept low so the terminal stays readable.",
    )
    args = parser.parse_args()

    reader = LiveReader(leader_port=args.leader_port, use_follower=False)
    print("[INFO] Connected to leader. Move the arm to the desired rest pose.")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        while True:
            joint_pos = reader.read().joint_pos
            values = ", ".join(
                f'"{n}": {v:+.4f}'
                for n, v in zip(_JOINT_NAMES, joint_pos[:6], strict=True)
            )
            print(f"\r{{{values}}}", end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
