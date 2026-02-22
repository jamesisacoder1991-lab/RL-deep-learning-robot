# RL-deep-learning-robot

This repo is a **single-file** RL workflow built for first-time setup with minimal manual debugging.

Main script: `all_in_one_rlbot_gym.py`

## What this gives you

- Auto-install + upgrade of required Python packages.
- PPO training with stronger defaults and checkpoint recovery.
- Auto-resume from latest checkpoint or existing model.
- Emergency saves on crash and snapshot save on keyboard interrupt.
- RLBot `agent.py` + config generation for private matches.
- Click-to-run Windows launchers for background training.
- Zip packaging for easy sharing.

---

## First test (recommended exact order)

### 1) Preflight everything

```bash
python all_in_one_rlbot_gym.py doctor
```

This command:
- installs packages,
- builds starter RLBot files,
- creates Windows launchers,
- smoke-checks environment creation.

### 2) Start first training run

```bash
python all_in_one_rlbot_gym.py train --timesteps 2000000 --model out/ppo_rlgym
```

### 3) Generate RLBot files (if you retrained with another model path)

```bash
python all_in_one_rlbot_gym.py rlbot-config --model out/ppo_rlgym.zip
```

### 4) Zip project/output for backup/share

```bash
python all_in_one_rlbot_gym.py zip --source . --name rocket_rl_all_in_one.zip
```

---

## No terminal / click-to-start (Windows)

Generate launchers:

```bash
python all_in_one_rlbot_gym.py make-launchers
```

Files created:
- `out/launchers/start_training.bat`
- `out/launchers/start_training_silent.vbs`

Double-click `start_training_silent.vbs` to run training in the background.

---

## Rocket League private/offline match setup

1. Install RLBot GUI on the PC with Rocket League.
2. Run:
   ```bash
   python all_in_one_rlbot_gym.py rlbot-config --model out/ppo_rlgym.zip
   ```
3. In RLBot GUI, load `out/rlbot_bot/agent.cfg`.
4. Launch Rocket League from RLBot GUI.
5. Create private/offline match and add the bot.
6. Iterate for stronger play:
   - train longer,
   - improve reward design,
   - improve observation/action mapping in generated `out/rlbot_bot/agent.py`.

---

## Safety / recovery files

- `out/checkpoints/` → periodic checkpoints.
- `out/best_model/` → best eval model.
- `out/ppo_rlgym_emergency.zip` → saved if training crashes.
- `out/ppo_rlgym_interrupted.zip` → saved if Ctrl+C interrupt.
- `out/training_config.json` → exact train settings.
- `out/logs/run.log` → runtime logs.

---

## Practical honesty

No script can guarantee “never fail no matter what” across all PCs and package ecosystems.
This setup is tuned to reduce failures and debugging workload as much as possible for a first run.
For best in-game strength, keep iterating reward/obs/action design and run long training.
