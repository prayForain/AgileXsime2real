#!/usr/bin/env python3
"""Read-only health check script for AgileX PiperX over SocketCAN.

Never enables or moves any motor. Safely reads firmware version, arm status,
joint angles, flange pose, motor/driver diagnostics, limits, and optional gripper status.
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


def wait_for(fn, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = fn()
        if res is not None:
            return res
        time.sleep(interval)
    return None


def main():
    parser = argparse.ArgumentParser(description="PiperX read-only health check")
    parser.add_argument("--channel", default="can0", help="SocketCAN channel (default: can0)")
    args = parser.parse_args()

    # Step 1: Connect with default config to read firmware version
    base_cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(base_cfg)
    robot.connect()

    fw_info = wait_for(robot.get_firmware, timeout=5.0)
    if fw_info is None:
        robot.disconnect()
        raise SystemExit(f"ERROR: Timed out waiting for firmware info on {args.channel}.")

    sw_ver = fw_info.get("software_version", "unknown")
    fw_profile = resolve_firmware_profile(ArmModel.PIPER_X, sw_ver)
    robot.disconnect()

    # Step 2: Reconnect using resolved firmware profile
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=fw_profile,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    try:
        # Give background reader a moment to populate all feedback frames
        time.sleep(0.6)

        arm_status = wait_for(robot.get_arm_status, timeout=3.0)
        joints = wait_for(robot.get_joint_angles, timeout=3.0)
        flange = wait_for(robot.get_flange_pose, timeout=3.0)
        motor_states = [wait_for(lambda idx=i: robot.get_motor_states(idx), timeout=2.0) for i in range(1, 7)]
        driver_states = [wait_for(lambda idx=i: robot.get_driver_states(idx), timeout=2.0) for i in range(1, 7)]
        angle_vel_limits = [wait_for(lambda idx=i: robot.get_joint_angle_vel_limits(idx), timeout=2.0) for i in range(1, 7)]
        acc_limits = [wait_for(lambda idx=i: robot.get_joint_acc_limits(idx), timeout=2.0) for i in range(1, 7)]
        crash_rating = wait_for(robot.get_crash_protection_rating, timeout=2.0)
        gripper_status = wait_for(effector.get_gripper_status, timeout=1.0)

        print("=== 1. Firmware & Connection ===")
        print(f"Channel          : {args.channel}")
        print(f"Software Version : {sw_ver}  (SDK profile: {fw_profile})")
        print(f"Hardware Version : {fw_info.get('hardware_version', 'N/A')}")
        print(f"Production Date  : {fw_info.get('production_date', 'N/A')}")
        print(f"Comm OK / FPS    : {robot.is_ok()} / {robot.get_fps():.1f} Hz")

        print("\n=== 2. Arm Status ===")
        if arm_status is not None:
            s = arm_status.msg
            print(f"ctrl_mode     : {s.ctrl_mode}")
            print(f"arm_status    : {s.arm_status}")
            print(f"mode_feedback : {s.mode_feedback}")
            print(f"teach_status  : {getattr(s, 'teach_status', 'N/A')}")
            print(f"motion_status : {s.motion_status}")
            err = s.err_status
            comm_errs = [getattr(err, f"communication_status_joint_{i}") for i in range(1, 7)]
            limit_errs = [getattr(err, f"joint_{i}_angle_limit") for i in range(1, 7)]
            print(f"comm_err J1-6 : {comm_errs}")
            print(f"limit_err J1-6: {limit_errs}")
        else:
            print("Arm status unavailable.")

        print("\n=== 3. Joint Angles & Motor/Driver Diagnostics ===")
        header = f"{'Joint':<6} {'Pos(deg)':>9} {'Pos(rad)':>9} {'Vel(r/s)':>9} {'Volt(V)':>8} {'FocT(C)':>8} {'MotT(C)':>8} {'En':>6} {'Err':>6} {'Lim(deg)':>18} {'ColLvl':>7}"
        print(header)
        print("-" * len(header))
        for i in range(6):
            pos_rad = joints.msg[i] if joints is not None else float("nan")
            pos_deg = math.degrees(pos_rad) if joints is not None else float("nan")
            vel = motor_states[i].msg.velocity if motor_states[i] is not None else float("nan")
            ds = driver_states[i].msg if driver_states[i] is not None else None
            vol = ds.vol if ds is not None else float("nan")
            foc_t = ds.foc_temp if ds is not None else float("nan")
            mot_t = ds.motor_temp if ds is not None else float("nan")
            en = ds.foc_status.driver_enable_status if ds is not None else None
            has_err = (
                any([
                    ds.foc_status.voltage_too_low,
                    ds.foc_status.motor_overheating,
                    ds.foc_status.driver_overcurrent,
                    ds.foc_status.driver_overheating if hasattr(ds.foc_status, "driver_overheating") else False,
                    ds.foc_status.collision_status,
                    ds.foc_status.driver_error_status,
                    ds.foc_status.stall_status,
                ])
                if ds is not None
                else None
            )
            if angle_vel_limits[i] is not None:
                lo_d = math.degrees(angle_vel_limits[i].msg.min_angle_limit)
                hi_d = math.degrees(angle_vel_limits[i].msg.max_angle_limit)
                lim_str = f"[{lo_d:6.1f}, {hi_d:6.1f}]"
            else:
                lim_str = "N/A"
            col_lvl = str(crash_rating.msg[i]) if crash_rating is not None else "N/A"
            print(
                f"J{i+1:<5} {pos_deg:9.2f} {pos_rad:9.4f} {vel:9.3f} {vol:8.1f} {foc_t:8.1f} {mot_t:8.1f} "
                f"{str(en):>6} {str(has_err):>6} {lim_str:>18} {col_lvl:>7}"
            )

        print("\n=== 4. Flange Pose ===")
        if flange is not None:
            x, y, z, rx, ry, rz = flange.msg
            print(f"XYZ (m)   : [{x:.4f}, {y:.4f}, {z:.4f}]")
            print(f"RPY (deg) : [{math.degrees(rx):.2f}, {math.degrees(ry):.2f}, {math.degrees(rz):.2f}]")
        else:
            print("Flange pose unavailable.")

        print("\n=== 5. Gripper Status ===")
        if gripper_status is not None:
            print(gripper_status.msg)
        else:
            print("No gripper feedback received (gripper not attached or offline).")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
