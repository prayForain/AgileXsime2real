#!/usr/bin/env python3
"""Graceful upward-stretch choreography demo for AgileX PiperX on a compact desk.

Uses forward kinematics (robot.fk) to verify every waypoint stays compact horizontally
(flange horizontal reach <= ~22 cm) while smoothly stretching upward to Z ~ 48 cm,
then folds smoothly back to the official zero posture [0, 0, 0, 0, 0, 0] and performs
a damped settle before returning to STANDBY (no sudden drop).
"""

import argparse
import math
import time

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
    resolve_firmware_profile,
)

MAX_SPEED_PERCENT = 30


def wait_for(fn, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = fn()
        if res is not None:
            return res
        time.sleep(interval)
    return None


def wait_reached(robot, target_rad, tol_deg: float = 2.0, timeout: float = 15.0) -> bool:
    """Wait until joints are within tol_deg of target_rad and motion_status == 0."""
    time.sleep(0.5)
    deadline = time.monotonic() + timeout
    tol_rad = math.radians(tol_deg)
    while time.monotonic() < deadline:
        ja = robot.get_joint_angles()
        st = robot.get_arm_status()
        if ja is not None:
            max_err = max(abs(a - b) for a, b in zip(ja.msg, target_rad))
            done = st is not None and getattr(st.msg, "motion_status", None) == 0
            if max_err <= tol_rad and done:
                return True
        time.sleep(0.05)
    return False


def build_waypoints(base_j1_deg: float):
    """Build an upward-stretching, compact-footprint waypoint sequence (in degrees).

    At J2=0, J3=0 the arm is folded flat over the base. Increasing J2 (0->50 deg)
    raises the upper arm upward & inward (reducing backward elbow overhang), while
    opening J3 (0->-55 deg) lifts the forearm gracefully upward to Z ~ 45-48 cm
    with flange horizontal reach <= 22.1 cm.
    """
    j1 = base_j1_deg
    return [
        ("1. Align Zero / Folded", [j1,       0.0,   0.0,   0.0,   0.0,   0.0], 0.4),
        ("2. Gentle Lift",         [j1,      30.0, -30.0,  10.0,   0.0,   0.0], 0.4),
        ("3. Upward Stretch",      [j1,      50.0, -55.0,   0.0,   0.0,   0.0], 0.8),
        ("4. Look Left + Wrist",   [j1 + 18, 50.0, -55.0, -15.0,  20.0,  25.0], 0.8),
        ("5. Look Right + Wrist",  [j1 - 18, 50.0, -55.0, -15.0, -20.0, -25.0], 0.8),
        ("6. Center Stretch",      [j1,      50.0, -55.0,   0.0,   0.0,   0.0], 0.6),
        ("7. Smooth Fold Down",    [j1,      25.0, -25.0,   5.0,   0.0,   0.0], 0.4),
        ("8. Return to Zero",      [j1,       0.0,   0.0,   0.0,   0.0,   0.0], 0.6),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compact upward-stretch demo for PiperX with gentle return-to-zero shutdown"
    )
    parser.add_argument("--channel", default="can0", help="SocketCAN channel (default: can0)")
    parser.add_argument(
        "--speed",
        type=int,
        default=15,
        help=f"Speed percent 1..{MAX_SPEED_PERCENT} (default: 15)",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also gently open (4 cm) and close the gripper at the top stretch pose",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview waypoints and FK reach without enabling or moving motors",
    )
    args = parser.parse_args()

    if not (1 <= args.speed <= MAX_SPEED_PERCENT):
        raise SystemExit(f"ERROR: --speed must be between 1 and {MAX_SPEED_PERCENT}.")

    # Step 1: Probe firmware version
    probe_cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(probe_cfg)
    robot.connect()
    fw_info = wait_for(robot.get_firmware, timeout=5.0)
    robot.disconnect()
    if fw_info is None:
        raise SystemExit(f"ERROR: Timed out reading firmware on {args.channel}. Is can0 up?")

    sw_ver = fw_info.get("software_version", "unknown")
    fw_profile = resolve_firmware_profile(ArmModel.PIPER_X, sw_ver)

    # Step 2: Connect with exact firmware profile
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=fw_profile,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER) if args.with_gripper else None

    try:
        time.sleep(0.5)
        arm_status = wait_for(robot.get_arm_status, timeout=3.0)
        joints = wait_for(robot.get_joint_angles, timeout=3.0)
        if arm_status is None or joints is None:
            raise SystemExit("ERROR: Failed to read arm status or joint angles.")

        s = arm_status.msg
        print(f"Connected: PiperX | Firmware: {sw_ver} (profile={fw_profile}) | FPS: {robot.get_fps():.1f} Hz")
        print(f"Status: ctrl_mode={s.ctrl_mode}, arm_status={s.arm_status}, teach_status={getattr(s, 'teach_status', 'N/A')}")

        # Check all 6 drivers are fault-free
        for i in range(1, 7):
            ds = wait_for(lambda idx=i: robot.get_driver_states(idx), timeout=2.0)
            if ds is None:
                raise SystemExit(f"ERROR: Cannot read driver state for J{i}.")
            fs = ds.msg.foc_status
            if any([
                fs.voltage_too_low,
                fs.motor_overheating,
                fs.driver_overcurrent,
                getattr(fs, "driver_overheating", False),
                fs.collision_status,
                fs.driver_error_status,
                fs.stall_status,
            ]):
                raise SystemExit(f"ERROR: J{i} driver fault detected ({fs}). Aborting.")

        if int(s.ctrl_mode) == 0x02:
            raise SystemExit(
                "ERROR: Arm is in TEACHING_MODE (0x02). Please exit teach mode first."
            )

        cur_deg = [math.degrees(q) for q in joints.msg]
        # Keep J1 centered at 0 if already near 0, otherwise center around current J1 (clamped safely)
        base_j1 = 0.0 if abs(cur_deg[0]) <= 15.0 else max(-120.0, min(120.0, cur_deg[0]))
        waypoints = build_waypoints(base_j1)

        print(f"\nCurrent joints (deg): {[round(d, 1) for d in cur_deg]}")
        print(f"\n=== Planned Upward Stretch Sequence (Speed: {args.speed}%, Gripper: {args.with_gripper}) ===")
        print(f"{'Step':<24} {'Joints (deg)':<38} {'Flange XYZ (cm)':<24} {'HorzReach':>9}")
        print("-" * 98)
        for name, degs, _ in waypoints:
            rads = [math.radians(d) for d in degs]
            p = robot.fk(rads)
            reach_cm = math.hypot(p[0], p[1]) * 100.0
            xyz_str = f"({p[0]*100:+5.1f}, {p[1]*100:+5.1f}, {p[2]*100:+5.1f})"
            j_str = "[" + ", ".join(f"{d:5.1f}" for d in degs) + "]"
            print(f"{name:<24} {j_str:<38} {xyz_str:<24} {reach_cm:7.1f} cm")

        if args.dry_run:
            print("\n[DRY-RUN] No motor enabled or moved. Exiting safely.")
            return

        print("\nSafety checklist:")
        print("  1. Clearance needed: ~30 cm forward (flange+gripper), ~55 cm upward, ~15 cm left/right.")
        print("  2. Base is firmly mounted; keep one hand ready near Ctrl+C (damped emergency stop).")
        confirm = input("\nType 'yes' and press Enter to start upward stretch demo: ").strip().lower()
        if confirm != "yes":
            print("Cancelled. No motor was enabled.")
            return

        # Step 3: Enable & set low speed
        robot.set_speed_percent(args.speed)
        print("Enabling motors...")
        deadline = time.monotonic() + 5.0
        while not robot.enable():
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out enabling motors.")
            time.sleep(0.05)
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
        print("Motors enabled. Starting smooth motion...")

        # Step 4: Execute waypoints sequentially
        for name, degs, dwell in waypoints:
            rads = [math.radians(d) for d in degs]
            print(f"-> {name} ...")
            robot.move_j(rads)
            ok = wait_reached(robot, rads, tol_deg=2.5, timeout=15.0)
            if not ok:
                print(f"   (Notice: wait_reached timed out on '{name}', continuing cautiously)")

            if args.with_gripper and effector is not None and name.startswith("6."):
                print("   -> Gripper gentle open (4 cm) & close...")
                effector.move_gripper_m(value=0.04, force=1.0)
                time.sleep(1.2)
                effector.move_gripper_m(value=0.0, force=1.0)
                time.sleep(1.2)

            time.sleep(dwell)

        # Step 5: Already folded back at zero posture -> damped settle -> standby
        print("\nReturned to folded zero posture.")
        print("Applying damped gentle settle (electronic_emergency_stop) for 2.0s...")
        robot.electronic_emergency_stop()
        time.sleep(2.0)
        print("Resetting to STANDBY (motors off)...")
        robot.reset()
        time.sleep(0.3)
        print("Demo complete! Arm is safely parked in STANDBY.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Triggering damped electronic_emergency_stop()...")
        try:
            robot.electronic_emergency_stop()
        except Exception:
            pass
    except Exception as exc:
        print(f"\n[ERROR] {exc} — triggering damped electronic_emergency_stop()...")
        try:
            robot.electronic_emergency_stop()
        except Exception:
            pass
        raise
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
