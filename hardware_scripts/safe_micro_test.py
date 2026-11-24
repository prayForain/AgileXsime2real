#!/usr/bin/env python3
"""Safe low-speed, small-amplitude single-joint motion test for AgileX PiperX.

Safety guards built in:
  - Defaults to J6 (wrist roll) by +10 deg at 10% speed (rotates flange in place,
    zero sweeping over a small desk).
  - Hard clamps: speed in [1, 20]%, |delta-deg| <= 15 deg, target within soft limits.
  - Prints exact start/target angles and requires typing 'yes' before enabling motors.
  - Ctrl+C immediately calls electronic_emergency_stop() (damped slow stop).
  - Always returns to the initial safe posture before prompting to disable motors.
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

MAX_SPEED_PERCENT = 20
MAX_DELTA_DEG = 15.0
LIMIT_MARGIN_DEG = 2.0


def wait_for(fn, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = fn()
        if res is not None:
            return res
        time.sleep(interval)
    return None


def wait_reached(robot, target_rad, tol_deg: float = 1.5, timeout: float = 10.0) -> bool:
    """Wait until joints are close to target_rad and motion_status reports done."""
    time.sleep(0.6)
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


def main():
    parser = argparse.ArgumentParser(
        description="Safe low-speed micro-motion test for PiperX (ideal for small desks)"
    )
    parser.add_argument("--channel", default="can0", help="SocketCAN channel (default: can0)")
    parser.add_argument(
        "--joint",
        type=int,
        default=6,
        choices=[1, 2, 3, 4, 5, 6],
        help="Single joint to test (default: 6 = wrist flange rotation in place)",
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=10.0,
        help=f"Angular offset in degrees (hard-capped at +/-{MAX_DELTA_DEG} deg, default: 10.0)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help=f"Speed percent 1..{MAX_SPEED_PERCENT} (default: 10)",
    )
    parser.add_argument(
        "--return-zero",
        action="store_true",
        help="After testing, slowly return to the official all-zero posture [0,0,0,0,0,0] "
        "(Note: will also rotate J1 to 0 deg; ensure desk clearance).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect state and planned angles only; do NOT enable or move motors",
    )
    args = parser.parse_args()

    if not (1 <= args.speed <= MAX_SPEED_PERCENT):
        raise SystemExit(f"ERROR: --speed must be between 1 and {MAX_SPEED_PERCENT} for this safe test.")
    if abs(args.delta_deg) > MAX_DELTA_DEG or args.delta_deg == 0:
        raise SystemExit(f"ERROR: --delta-deg must be non-zero and within +/-{MAX_DELTA_DEG} deg.")

    # Step 1: Detect firmware version
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

    try:
        time.sleep(0.5)
        arm_status = wait_for(robot.get_arm_status, timeout=3.0)
        joints = wait_for(robot.get_joint_angles, timeout=3.0)
        if arm_status is None or joints is None:
            raise SystemExit("ERROR: Failed to read arm status or joint angles.")

        s = arm_status.msg
        print(f"Connected: PiperX | Firmware: {sw_ver} (profile={fw_profile}) | FPS: {robot.get_fps():.1f} Hz")
        print(f"Arm status: ctrl_mode={s.ctrl_mode}, arm_status={s.arm_status}, teach_status={getattr(s, 'teach_status', 'N/A')}")

        # Abort if any communication or driver fault exists
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
                raise SystemExit(f"ERROR: J{i} driver reported a fault ({fs}). Aborting.")

        if int(s.ctrl_mode) == 0x02:
            raise SystemExit(
                "ERROR: Arm is currently in TEACHING_MODE (0x02). "
                "Please ensure the teach LED on J5 is OFF and switch to standby before running CAN motion."
            )

        # Compute clamped safe start and single-joint micro target
        limits_rad = [cfg["joint_limits"][f"joint{i}"] for i in range(1, 7)]
        cur_rad = list(joints.msg)
        start_safe_rad = [
            max(lo, min(hi, q)) for q, (lo, hi) in zip(cur_rad, limits_rad)
        ]

        idx = args.joint - 1
        lo, hi = limits_rad[idx]
        margin = math.radians(LIMIT_MARGIN_DEG)
        desired = start_safe_rad[idx] + math.radians(args.delta_deg)
        # If moving in requested direction hits limit, flip direction automatically
        if desired > hi - margin or desired < lo + margin:
            desired = start_safe_rad[idx] - math.radians(args.delta_deg)
        desired = max(lo + margin, min(hi - margin, desired))

        target_rad = list(start_safe_rad)
        target_rad[idx] = desired

        print(f"\n=== Planned Micro-Motion (Speed: {args.speed}%, Joint: J{args.joint}) ===")
        print(f"{'Joint':<6} {'Current(deg)':>13} {'SafeStart(deg)':>15} {'Target(deg)':>12} {'Delta(deg)':>11}")
        print("-" * 62)
        for i in range(6):
            c_d = math.degrees(cur_rad[i])
            s_d = math.degrees(start_safe_rad[i])
            t_d = math.degrees(target_rad[i])
            d_d = t_d - s_d
            mark = "  <-- test joint" if i == idx else ""
            print(f"J{i+1:<5} {c_d:13.2f} {s_d:15.2f} {t_d:12.2f} {d_d:+11.2f}{mark}")

        max_init_shift = max(abs(math.degrees(a - b)) for a, b in zip(cur_rad, start_safe_rad))
        if max_init_shift > 0.5:
            print(
                f"\nNote: Some resting joints are slightly outside soft limits (max {max_init_shift:.2f} deg); "
                "the arm will first gently align into SafeStart."
            )

        if args.dry_run:
            print("\n[DRY-RUN] No motor enabled or moved. Exiting safely.")
            return

        print("\nSafety checklist before proceeding:")
        print("  1. Base is firmly mounted on the desk; hands & objects are clear.")
        print("  2. Keep one hand ready near Ctrl+C (triggers damped emergency stop).")
        confirm = input("\nType 'yes' and press Enter to enable & run this micro-motion: ").strip().lower()
        if confirm != "yes":
            print("Cancelled. No motor was enabled.")
            return

        # Step 3: Set low speed & enable
        robot.set_speed_percent(args.speed)
        print("Enabling motors...")
        enable_deadline = time.monotonic() + 5.0
        while not robot.enable():
            if time.monotonic() > enable_deadline:
                raise RuntimeError("Timed out enabling motors.")
            time.sleep(0.05)
        print("Motors enabled.")

        # Step 4: Hold SafeStart -> move to Target -> return to SafeStart
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)

        print("1/3 Moving gently to SafeStart...")
        robot.move_j(start_safe_rad)
        wait_reached(robot, start_safe_rad, timeout=8.0)
        time.sleep(0.5)

        print(f"2/3 Moving J{args.joint} to Target ({math.degrees(target_rad[idx]):.2f} deg)...")
        robot.move_j(target_rad)
        reached = wait_reached(robot, target_rad, timeout=10.0)
        ja_mid = robot.get_joint_angles()
        if ja_mid is not None:
            print(f"    Reached J{args.joint} = {math.degrees(ja_mid.msg[idx]):.2f} deg (ok={reached})")
        time.sleep(1.0)

        print(f"3/3 Returning J{args.joint} back to SafeStart ({math.degrees(start_safe_rad[idx]):.2f} deg)...")
        robot.move_j(start_safe_rad)
        reached_back = wait_reached(robot, start_safe_rad, timeout=10.0)
        ja_end = robot.get_joint_angles()
        if ja_end is not None:
            print(f"    Returned J{args.joint} = {math.degrees(ja_end.msg[idx]):.2f} deg (ok={reached_back})")

        print("\nMicro-motion test finished successfully!")
        j1_deg = math.degrees(start_safe_rad[0])
        if args.return_zero:
            choice = "zero"
        else:
            print("Choose how to park and power down motors (no sudden drop):")
            print("  [Enter] Local fold + damped settle (keeps J1 at current angle, safest on small desk)")
            print(f"  [zero]  Slowly move to official all-zero [0,0,0,0,0,0] (NOTE: J1 will rotate {abs(j1_deg):.1f} deg!) + damped settle")
            print("  [keep]  Stay enabled at current pose")
            choice = input("Your choice [Enter / zero / keep]: ").strip().lower()

        if choice == "zero":
            if not args.return_zero and abs(j1_deg) > 15.0:
                ok_z = input(
                    f"Confirm desk clearance for J1 rotating {j1_deg:+.1f} deg -> 0.0 deg (type 'yes'): "
                ).strip().lower()
                if ok_z != "yes":
                    print("Skipped all-zero rotation; falling back to local fold + damped settle.")
                    choice = ""
            if choice == "zero":
                zero_rad = [0.0] * 6
                print(f"Moving slowly ({args.speed}%) to official zero posture [0, 0, 0, 0, 0, 0]...")
                robot.move_j(zero_rad)
                wait_reached(robot, zero_rad, timeout=15.0)
                time.sleep(0.5)

        if choice != "keep":
            if choice != "zero":
                # Ensure J2/J3 are folded down to 0 deg in place (keeping current J1)
                local_fold_rad = list(start_safe_rad)
                local_fold_rad[1] = 0.0
                local_fold_rad[2] = 0.0
                print("Ensuring folded low posture (J2=0, J3=0, keeping current J1)...")
                robot.move_j(local_fold_rad)
                wait_reached(robot, local_fold_rad, timeout=10.0)
                time.sleep(0.5)

            # Official gentle shutdown: damped settle -> wait -> reset to standby
            print("Applying damped gentle settle (electronic_emergency_stop) for 2.0s...")
            robot.electronic_emergency_stop()
            time.sleep(2.0)
            print("Resetting to STANDBY (motors off)...")
            robot.reset()
            time.sleep(0.3)
            print("Parked safely in STANDBY with motors disabled.")
        else:
            print("Motors left ENABLED as requested.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Triggering electronic_emergency_stop()...")
        try:
            robot.electronic_emergency_stop()
        except Exception:
            pass
    except Exception as exc:
        print(f"\n[ERROR] {exc} — triggering electronic_emergency_stop()...")
        try:
            robot.electronic_emergency_stop()
        except Exception:
            pass
        raise
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
