# ROB6323 – Unitree Go2 RL Locomotion (Isaac Lab)

This repository contains a **custom reinforcement-learning locomotion task** for the Unitree Go2 quadruped, implemented using **Isaac Lab** and **RSL-RL (PPO)**.

The final trained policy demonstrates:
- Stable forward walking
- **Clean and accurate side-stepping (lateral motion)**
- Minimal foot scuffing
- Stable posture (no crouching or knee collapse)
- Smooth, low-torque joint behavior

All modifications strictly follow the project rules.

---

## 1. Files Modified (and Only These)

Per the project instructions, **only the following two files were edited**:

- `rob6323_go2_env_cfg.py`
- `rob6323_go2_env.py`

No other files, PPO hyperparameters, imports, or infrastructure were modified.

---

## 2. Major Additions and Rationale

### 2.1 Explicit Torque-Controlled PD Actuation
**Location:** `rob6323_go2_env.py`

Instead of relying on implicit simulator PD gains, joint torques are explicitly computed:

\[
\tau = K_p (q_{des} - q) - K_d \dot{q}
\]

**Why**
- Makes torque penalties physically meaningful
- Prevents hidden actuator stiffness exploitation
- Improves stability and realism

---

### 2.2 Torque Regularization Reward
**Reward term:** `torque`

Penalizes squared applied joint torques (after clipping).

**Why**
- Encourages energy-efficient motion
- Reduces violent joint accelerations
- Produces smoother gaits

---

### 2.3 Action Rate & Smoothness Penalty
**Reward term:** `rew_action_rate`

Penalizes:
- Action rate (Δa)
- Action acceleration (Δ²a)

**Why**
- Suppresses high-frequency oscillations
- Eliminates hopping and jitter
- Improves PPO convergence stability

---

### 2.4 Foot Scuffing Reduction
**Reward terms:**
- `feet_clearance`
- Contact-based penalties during swing

**Why**
- Prevents dragging feet
- Encourages clean swing phases
- Improves realism and reduces ground scraping

---

### 2.5 Posture & Stability Regularization
**Reward terms:**
- `base_height`
- `orient`
- `lin_vel_z`
- `ang_vel_xy`
- `dof_pos`
- `dof_vel`

**Why**
- Prevents crouching / knee collapse
- Maintains upright torso
- Reduces bouncing and roll/pitch oscillations

---

### 2.6 Raibert Foot Placement Heuristic
**Reward term:** `raibert_heuristic`

Encourages foot placements consistent with commanded velocity.

**Why**
- Improves leg coordination
- Stabilizes stepping geometry
- Especially helpful for lateral and diagonal motion

---

## 3. Side-Stepping-Specific Reward Tuning (Tier-A Safe Changes)

After achieving stable forward walking, the reward was tuned specifically for **lateral motion**.

### Changes Applied

| Reward Term | Change | Purpose |
|-----------|-------|--------|
| `track_lin_vel_xy_exp` | × **1.25** | Strongly rewards lateral velocity tracking |
| `track_ang_vel_z_exp` | × **1.10** | Reduces yaw drift during side-stepping |
| Linear velocity tracking sigma | × **0.85** | Makes directional error more costly |

**Why this works**
- Side-stepping requires accurate **vy tracking**
- Small yaw errors otherwise leak into xy motion (“crabbing”)
- Tightening the exponential tracking sharpens directional accuracy without destabilizing training

---

## 4. Observations and Command Space

### Observations
- Base linear velocity (body frame)
- Base angular velocity
- Projected gravity
- Joint positions and velocities
- Previous actions
- Gait phase signals (sin / cos clocks)

### Commands
- `v_x`: forward velocity
- `v_y`: lateral velocity (side-stepping)
- `ω_z`: yaw rate

Commands are randomly sampled at reset.

---

## 5. Training Setup

- **Algorithm:** PPO (RSL-RL)
- **Simulation timestep:** 200 Hz
- **Control decimation:** 4
- **Parallel environments:** 4096
- **Episode length:** 20 seconds
- **PPO hyperparameters:** unchanged

---

## 6. How to Reproduce (Exact Commands)

### 6.1 Activate Isaac Lab Environment
```bash
source isaaclab/setup_conda_env.sh

### 6.2 Training Command (Primary Run)
python source/isaaclab_tasks/isaaclab_tasks/scripts/rsl_rl/train.py \
  task=rob6323_go2_flat_direct \
  seed=133581 \
  headless=True


### 6.3 Video Visualization
python source/isaaclab_tasks/isaaclab_tasks/scripts/rsl_rl/train.py \
  task=rob6323_go2_flat_direct \
  seed=133581 \
  video=True
