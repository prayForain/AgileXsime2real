#!/usr/bin/env python3
"""Safe Real-Robot Sim-to-Real deployment script for fine-tuned SmolVLA on AgileX PiperX.

Features:
  - 5-layer physical safety guardrails (--dry-run, 'yes' confirm, 3.5 deg/step clamp,
    robot.fk() min_flange_z anti-table floor guard, Ctrl+C damped e-stop).
  - Exponential moving average (EMA) smoothing to eliminate chunk-boundary jitter.
  - Automatic lift-success early stop: once grasped & lifted steadily for 1.2s, stops
    inference immediately (no prolonged hovering jitter).
  - Graceful place-down: softly sets the foam back on the desk, opens gripper, then
    folds back to zero + damped settle -> STANDBY.
"""

import argparse
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
    resolve_firmware_profile,
)

SIM2REAL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WRIST_CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1Z8620073-video-index0"
DEFAULT_GLOBAL_CAM = "/dev/v4l/by-id/usb-Orbbec_R__Orbbec_Gemini_336L_CPCA9630003X-video-index0"
DEFAULT_CKPT = str(SIM2REAL_ROOT / "outputs" / "smolvla_pick_foam_aligned" / "checkpoints" / "005000" / "pretrained_model")
READY_ARM_RAD = [0.0, 0.70, -0.85, 0.70, 0.0, 0.0]


class ThreadedCamera:
    """Continuously drains V4L2 frames on a background thread so UVC cameras
    (especially Dabai DC1) never stall with select() timeout during pauses AND
    the policy always receives the freshest zero-latency RGB frame."""

    def __init__(self, dev_str: str, width: int = 320, height: int = 240, fps: int = 15):
        self.dev_str = dev_str
        self.width = width
        self.height = height
        dev = int(dev_str) if dev_str.isdigit() else dev_str
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {dev_str}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        self.release()
        raise RuntimeError(
            f"Camera {dev_str} opened but timed out waiting for first frame (try replugging its USB)."
        )

    def _reader(self):
        while self._running:
            ret, bgr = self.cap.read()
            if ret and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                    rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._frame = rgb
            else:
                time.sleep(0.02)

    def read_rgb(self) -> np.ndarray:
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"No frame available from {self.dev_str}")
            return self._frame.copy()

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


def wait_for(fn, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = fn()
        if val is not None:
            return val
        time.sleep(0.05)
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy fine-tuned SmolVLA onto real PiperX")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--channel", type=str, default="can0")
    parser.add_argument("--global-cam", type=str, default=DEFAULT_GLOBAL_CAM)
    parser.add_argument("--wrist-cam", type=str, default=DEFAULT_WRIST_CAM)
    parser.add_argument("--task", type=str, default="Pick up the black foam block.")
    parser.add_argument("--hz", type=float, default=10.0, help="Control loop rate in Hz (default: 10)")
    parser.add_argument("--speed", type=int, default=18, help="PiperX speed percent (max 30, default: 18)")
    parser.add_argument("--max-step-deg", type=float, default=3.5, help="Max joint change per step (deg)")
    parser.add_argument("--ema-alpha", type=float, default=1.0, help="Joint EMA weight (1.0 = off / no lag, recommended)")
    parser.add_argument("--min-flange-z", type=float, default=0.145, help="Min safe flange Z in meters (anti-table guard)")
    parser.add_argument("--lift-done-z", type=float, default=0.255, help="Flange Z (m) threshold to detect lift completion")
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true", help="Run live cameras + SmolVLA without enabling motors")
    args = parser.parse_args()

    if not (1 <= args.speed <= 30):
        raise SystemExit("ERROR: --speed must be between 1 and 30 for safe deployment.")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"ERROR: Checkpoint not found: {ckpt}")

    # 1. Load SmolVLA policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SmolVLA from {ckpt} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.reset()

    # 2. Open both cameras with background reader threads (prevents UVC select() timeout & stale frames)
    print(f"Opening global camera ({args.global_cam}) and wrist camera ({args.wrist_cam}) ...")
    cap_global = ThreadedCamera(args.global_cam, width=320, height=240, fps=30)
    cap_wrist = ThreadedCamera(args.wrist_cam, width=320, height=240, fps=15)

    # 3. Connect PiperX with auto-detected firmware
    probe = AgxArmFactory.create_arm(
        create_agx_arm_config(robot=ArmModel.PIPER_X, interface="socketcan", channel=args.channel)
    )
    probe.connect()
    fw = wait_for(probe.get_firmware, timeout=5.0)
    probe.disconnect()
    if fw is None:
        raise SystemExit(f"ERROR: Could not read PiperX firmware on {args.channel} (check `ip link show can0`)")
    fw_profile = resolve_firmware_profile(ArmModel.PIPER_X, fw["software_version"])

    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=fw_profile,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    time.sleep(0.5)

    limits = [cfg["joint_limits"][f"joint{i}"] for i in range(1, 7)]
    max_step_rad = math.radians(args.max_step_deg)
    alpha = max(0.1, min(1.0, args.ema_alpha))

    try:
        if not args.dry_run:
            print("\n=== REAL PIPERX VLA DEPLOYMENT SAFETY CHECK ===")
            print(f"Task         : {args.task}")
            print(f"Speed / Rate : {args.speed}% / {args.hz} Hz | Max step: {args.max_step_deg} deg | EMA: {alpha}")
            print(f"Min Flange Z : {args.min_flange_z*100:.1f} cm | Auto-stop Lift Z: {args.lift_done_z*100:.1f} cm")
            print("Keep one hand near Ctrl+C at all times (triggers damped emergency stop).")
            if input("Type 'yes' to enable motors and start: ").strip().lower() != "yes":
                print("Cancelled.")
                return

            robot.set_speed_percent(15)
            deadline = time.monotonic() + 5.0
            while not robot.enable():
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out enabling PiperX")
                time.sleep(0.05)
            robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)

            print("Moving smoothly to ready observation posture...")
            robot.move_j(READY_ARM_RAD)
            effector.move_gripper_m(value=0.09, force=1.0)
            time.sleep(3.5)
            robot.set_speed_percent(args.speed)

        dt = 1.0 / args.hz
        smoothed_q6 = None
        smoothed_grip = None
        ever_grasped_low = False
        grasp_q6 = None
        lifted_hold_steps = 0
        lift_hold_required = max(6, int(1.2 * args.hz))
        lifted_success = False

        print(f"Starting {'DRY-RUN' if args.dry_run else 'LIVE'} SmolVLA control loop (max {args.max_steps} steps)...")
        for step in range(args.max_steps):
            t0 = time.monotonic()
            cam1 = cap_global.read_rgb()
            cam2 = cap_wrist.read_rgb()

            ja = robot.get_joint_angles()
            gs = effector.get_gripper_status()
            if ja is None:
                continue
            cur_q6 = list(ja.msg)
            cur_grip = float(gs.msg.value) if gs is not None else 0.0
            cur_fk_z = robot.fk(cur_q6)[2]
            state7 = torch.tensor(cur_q6 + [cur_grip], dtype=torch.float32)

            obs = {
                "observation.state": state7,
                "observation.images.camera1": torch.from_numpy(cam1).permute(2, 0, 1).float() / 255.0,
                "observation.images.camera2": torch.from_numpy(cam2).permute(2, 0, 1).float() / 255.0,
                "task": args.task,
            }
            with torch.no_grad():
                batch = preprocessor(obs)
                act = postprocessor(policy.select_action(batch))
            pred = act.squeeze(0).detach().cpu().float().tolist()

            # Optional joint EMA (off by default at alpha=1.0 so there is zero phase lag).
            # NEVER apply EMA to the gripper so close/open commands execute crisply without delay!
            if smoothed_q6 is None or alpha >= 0.999:
                smoothed_q6 = list(pred[:6])
            else:
                smoothed_q6 = [alpha * pred[i] + (1.0 - alpha) * smoothed_q6[i] for i in range(6)]
            safe_grip = max(0.0, min(0.095, float(pred[6])))

            # Safety guardrail 1: clamp per-step delta and joint soft limits
            safe_q6 = []
            for i in range(6):
                lo, hi = limits[i]
                delta = max(-max_step_rad, min(max_step_rad, smoothed_q6[i] - cur_q6[i]))
                safe_q6.append(max(lo, min(hi, cur_q6[i] + delta)))

            # Safety guardrail 2: anti-table collision floor check
            fk_z = robot.fk(safe_q6)[2]
            if fk_z < args.min_flange_z:
                print(f"[GUARD step {step}] Blocked low flange Z={fk_z*100:.1f}cm < {args.min_flange_z*100:.1f}cm; holding pose.")
                safe_q6 = cur_q6

            if args.dry_run:
                if step % 5 == 0:
                    print(
                        f"[DRY step {step:3d}] cur_z={cur_fk_z*100:.1f}cm -> pred_z={fk_z*100:.1f}cm | "
                        f"grip={safe_grip*100:.1f}cm | pred_deg={[round(math.degrees(x),1) for x in safe_q6]}"
                    )
            else:
                robot.move_j(safe_q6)
                effector.move_gripper_m(value=safe_grip, force=1.5)

                # Track grasp-then-lift completion so we stop cleanly without hovering jitter
                if cur_fk_z < 0.23 and safe_grip < 0.045:
                    ever_grasped_low = True
                    grasp_q6 = list(cur_q6)

                if ever_grasped_low and cur_fk_z >= args.lift_done_z and safe_grip < 0.055:
                    lifted_hold_steps += 1
                    if lifted_hold_steps >= lift_hold_required:
                        lifted_success = True
                        print(
                            f"\n[SUCCESS] Step {step}: Foam grasped & lifted cleanly "
                            f"(flange Z={cur_fk_z*100:.1f} cm)! Stopping inference cleanly."
                        )
                        break
                else:
                    lifted_hold_steps = 0

            sleep_t = dt - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

        if not args.dry_run:
            robot.set_speed_percent(12)
            if lifted_success and grasp_q6 is not None:
                print("Holding lifted foam steadily for 1.5s...")
                time.sleep(1.5)
                print("Gently placing foam back down onto the desk...")
                robot.move_j(grasp_q6)
                time.sleep(2.5)
                print("Opening gripper to release foam...")
                effector.move_gripper_m(value=0.09, force=1.0)
                time.sleep(1.2)
                print("Lifting clear before folding...")
                robot.move_j(READY_ARM_RAD)
                time.sleep(2.0)

            print("Returning gently to folded zero posture...")
            robot.move_j([0.0] * 6)
            time.sleep(4.0)
            robot.electronic_emergency_stop()
            time.sleep(2.0)
            robot.reset()
            print("Parked safely in STANDBY.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Triggering damped electronic_emergency_stop()...")
        if not args.dry_run:
            robot.electronic_emergency_stop()
    finally:
        cap_global.release()
        cap_wrist.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
