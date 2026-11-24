# PiperX Sim-to-Real with ManiSkill 3 + LeRobot (SmolVLA)

End-to-end **Simulation-to-Real (Sim-to-Real)** manipulation pipeline for the **AgileX PiperX** 6-DoF robotic arm (with 2-finger `AgxGripper`, wrist `Orbbec Dabai DC1` RGB camera, and third-person `Orbbec Gemini 336L` RGB camera):

1. **ManiSkill 3** custom `PiperXAgent` + `PiperXPickFoam-v1` tabletop task (0.26 mm FK match with `pyAgxArm` SDK, aligned wrist & global cameras, visual domain randomization).
2. **Automatic expert data generation** using SAPIEN native `PinocchioModel` IK (no manual teleop required).
3. **LeRobot + SmolVLA** fine-tuning (`bf16`, ~4.2 GB VRAM on RTX 4080, ~19 min for 5,000 steps).
4. **Closed-loop simulation evaluation** (70% success rate on unseen randomized poses with 5,000 steps) + **guarded real-robot deployment** via `pyAgxArm` over SocketCAN.

---

## Directory Structure

```text
sim2real/
├── assets/
│   └── piper_x_with_gripper.urdf   # Standalone ManiSkill 3 URDF (references agx_arm_urdf meshes)
├── piperx_sim/
│   ├── piper_x_agent.py            # ManiSkill 3 BaseAgent (PiperX + AgxGripper + wrist_camera)
│   └── pick_foam_env.py            # PiperXPickFoam-v1 tabletop grasping env with domain randomization
├── scripts/
│   ├── verify_sim_piperx.py        # Cross-check sim FK vs pyAgxArm SDK & render camera views
│   ├── demo_pick_foam.py           # Single automatic IK grasp demo (storyboard + H.264 video)
│   ├── collect_sim_demos.py        # Batch collect N domain-randomized episodes (.npz)
│   ├── convert_to_lerobot.py       # Convert .npz episodes -> LeRobotDataset (camera1 + camera2)
│   ├── train_smolvla.sh            # Fine-tune lerobot/smolvla_base in bf16
│   ├── smolvla_policy_server.py    # Local TCP inference server (lerobot Python 3.12 env)
│   ├── eval_sim_smolvla.py         # Closed-loop ManiSkill 3 evaluation client (piperx_vla env)
│   └── deploy_real_piperx.py       # Guarded real PiperX deployment (ThreadedCamera + FK floor guard)
├── hardware_scripts/
│   ├── check_piperx.py             # Read-only PiperX firmware/status/motor/gripper diagnostics
│   ├── check_wrist_cam.py          # Quick OpenCV UVC camera snapshot check
│   ├── safe_micro_test.py          # Low-speed small-amplitude single-joint bring-up test
│   └── upward_stretch_demo.py      # Compact upward-stretch demo with damped return-to-zero
├── .gitignore
└── README.md
```

---

## 1. Environment Setup

We use two clean Conda environments to avoid Python/NumPy version conflicts between ManiSkill 3 and LeRobot 0.7.x (`requires-python >= 3.12`):

### 1.1 Clone Third-Party Dependencies

Inside this `sim2real/` directory:

```bash
# Official AgileX PiperX + AgxGripper URDF & meshes (required by assets/piper_x_with_gripper.urdf)
git clone --depth 1 https://github.com/agilexrobotics/agx_arm_urdf.git

# Official AgileX Python SDK (or reuse an existing clone)
git clone --depth 1 https://github.com/agilexrobotics/pyAgxArm.git

# Hugging Face LeRobot (v0.7+)
git clone --depth 1 https://github.com/huggingface/lerobot.git
```

### 1.2 Simulation Environment (`piperx_vla`, Python 3.11)

```bash
conda create -n piperx_vla python=3.11 -y
conda activate piperx_vla
pip install torch torchvision torchaudio mani_skill opencv-python
pip install -e ./pyAgxArm
```

### 1.3 LeRobot + SmolVLA + Real Deployment Environment (`lerobot`, Python 3.12)

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
pip install -e "./lerobot[smolvla,dataset]"
pip install -e ./pyAgxArm
```

---

## 2. Real Hardware Bring-Up (Ubuntu SocketCAN)

After powering on PiperX (24 V DC, green status LED blinking) and plugging in the USB-CAN adapter:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000

conda activate lerobot
# 1) Read-only health check (firmware, joint angles, voltages, temperatures, gripper)
python hardware_scripts/check_piperx.py

# 2) Optional safe micro-motion test (defaults to J6 +10 deg at 10% speed)
python hardware_scripts/safe_micro_test.py
```

---

## 3. Sim-to-Real Full Pipeline

### Step 1: Verify PiperX URDF & Kinematics in ManiSkill 3

```bash
conda activate piperx_vla
python scripts/verify_sim_piperx.py
python scripts/demo_pick_foam.py
```

### Step 2: Collect Domain-Randomized Simulation Demonstrations

```bash
conda activate piperx_vla
python scripts/collect_sim_demos.py \
  --num-episodes 150 \
  --out-dir data/raw_episodes_aligned \
  --seed 2000
```

### Step 3: Convert to `LeRobotDataset`

```bash
conda activate lerobot
python scripts/convert_to_lerobot.py \
  --raw-dir data/raw_episodes_aligned \
  --out-root data/piperx_pick_foam_aligned_lerobot \
  --repo-id local/piperx_pick_foam_aligned
```

### Step 4: Fine-Tune `SmolVLA` (RTX 4080 16 GB)

```bash
conda activate lerobot
bash scripts/train_smolvla.sh
```

### Step 5: Closed-Loop Simulation Evaluation

Terminal 1 (`lerobot` env — policy server):
```bash
conda activate lerobot
python scripts/smolvla_policy_server.py \
  --checkpoint outputs/smolvla_pick_foam_aligned/checkpoints/005000/pretrained_model \
  --port 8765
```

Terminal 2 (`piperx_vla` env — ManiSkill 3 rollout):
```bash
conda activate piperx_vla
python scripts/eval_sim_smolvla.py \
  --port 8765 \
  --num-episodes 10 \
  --domain-randomize
```

### Step 6: Real PiperX Deployment (Sim-to-Real)

Find your persistent camera paths with `ls -la /dev/v4l/by-id/`, then test first with `--dry-run` (motors disabled):

```bash
conda activate lerobot
python scripts/deploy_real_piperx.py \
  --global-cam /dev/v4l/by-id/<YOUR_GLOBAL_CAM> \
  --wrist-cam /dev/v4l/by-id/<YOUR_WRIST_CAM> \
  --dry-run
```

When verified, run live with safety guardrails (15% speed, 3.5 deg/step clamp, FK minimum Z height floor guard, Ctrl+C damped emergency stop):

```bash
python scripts/deploy_real_piperx.py \
  --global-cam /dev/v4l/by-id/<YOUR_GLOBAL_CAM> \
  --wrist-cam /dev/v4l/by-id/<YOUR_WRIST_CAM> \
  --speed 15 \
  --hz 10
```

---

## Safety Notes

- Never power PiperX above **26 V DC** (standard 24 V).
- Always keep the workspace clear and keep one hand ready on **Ctrl+C** (`electronic_emergency_stop()` triggers a smooth damped descent).
- Do not call `disable()` directly while the arm is raised; always return to the folded zero posture `[0, 0, 0, 0, 0, 0]` and perform a damped settle first.
