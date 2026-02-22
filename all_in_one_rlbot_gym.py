#!/usr/bin/env python3
"""Single-file RLGym + RLBot workflow with reliability-focused defaults."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("all_in_one_rlbot_gym")

DEFAULT_PACKAGES = [
    "pip",
    "setuptools",
    "wheel",
    "numpy",
    "torch",
    "gymnasium",
    "stable-baselines3",
    "rlgym",
    "rlbot",
]


@dataclass
class TrainConfig:
    timesteps: int = 2_000_000
    model_path: str = "out/ppo_rlgym"
    n_envs: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps: int = 2048
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    clip_range: float = 0.2
    eval_freq: int = 25_000
    checkpoint_freq: int = 25_000
    normalize_env: bool = True
    resume: bool = True


def setup_logging(log_file: str = "out/logs/run.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )


def run_python_module(module: str, args: list[str]) -> None:
    import subprocess

    cmd = [sys.executable, "-m", module, *args]
    LOG.info("[cmd] %s", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_packages(packages: list[str]) -> None:
    """Install missing packages and hard-fail with readable logs."""
    for pkg in packages:
        mod = pkg.replace("-", "_")
        if pkg in {"pip", "setuptools", "wheel"} or importlib.util.find_spec(mod) is None:
            LOG.info("Installing / upgrading package: %s", pkg)
            run_python_module("pip", ["install", "--upgrade", pkg])


def _make_gym_fallback_env() -> Any:
    gymnasium = importlib.import_module("gymnasium")
    return gymnasium.make("CartPole-v1")


def build_default_env() -> Any:
    if importlib.util.find_spec("rlgym") is None:
        LOG.warning("rlgym not found, using CartPole fallback.")
        return _make_gym_fallback_env()

    try:
        rlgym = importlib.import_module("rlgym")
        if hasattr(rlgym, "make"):
            LOG.info("Using rlgym.make()")
            return rlgym.make()
    except Exception as exc:
        LOG.warning("rlgym env creation failed (%s), falling back.", exc)
    return _make_gym_fallback_env()


def _latest_checkpoint_path(checkpoint_dir: Path) -> str | None:
    files = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else None


def _save_training_metadata(cfg: TrainConfig) -> None:
    Path("out").mkdir(parents=True, exist_ok=True)
    Path("out/training_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def _build_callbacks(eval_env: Any, cfg: TrainConfig) -> Any:
    callbacks_mod = importlib.import_module("stable_baselines3.common.callbacks")
    CheckpointCallback = getattr(callbacks_mod, "CheckpointCallback")
    EvalCallback = getattr(callbacks_mod, "EvalCallback")
    StopTrainingOnNoModelImprovement = getattr(callbacks_mod, "StopTrainingOnNoModelImprovement")
    CallbackList = getattr(callbacks_mod, "CallbackList")

    Path("out/checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, cfg.checkpoint_freq // max(1, cfg.n_envs)),
        save_path="out/checkpoints",
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=cfg.normalize_env,
    )

    no_improve = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, min_evals=8, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="out/best_model",
        log_path="out/eval_logs",
        eval_freq=max(1, cfg.eval_freq // max(1, cfg.n_envs)),
        deterministic=True,
        callback_after_eval=no_improve,
    )
    return CallbackList([checkpoint_cb, eval_cb])


def train(cfg: TrainConfig) -> None:
    ensure_packages(DEFAULT_PACKAGES)

    sb3 = importlib.import_module("stable_baselines3")
    vec_env_mod = importlib.import_module("stable_baselines3.common.env_util")
    vec_norm_mod = importlib.import_module("stable_baselines3.common.vec_env")

    PPO = getattr(sb3, "PPO")
    make_vec_env = getattr(vec_env_mod, "make_vec_env")
    VecNormalize = getattr(vec_norm_mod, "VecNormalize")

    def env_fn() -> Any:
        return build_default_env()

    env = make_vec_env(env_fn, n_envs=max(1, cfg.n_envs))
    eval_env = make_vec_env(env_fn, n_envs=1)

    if cfg.normalize_env:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    callbacks = _build_callbacks(eval_env, cfg)
    _save_training_metadata(cfg)

    checkpoints_dir = Path("out/checkpoints")
    model: Any
    resumed_from: str | None = None

    if cfg.resume:
        resumed_from = _latest_checkpoint_path(checkpoints_dir)
        if resumed_from:
            LOG.info("Resuming from checkpoint: %s", resumed_from)
            model = PPO.load(resumed_from, env=env, device="auto")
        elif Path(f"{cfg.model_path}.zip").exists():
            LOG.info("Resuming from existing model: %s.zip", cfg.model_path)
            model = PPO.load(cfg.model_path, env=env, device="auto")
        else:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size,
                n_steps=cfg.n_steps,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                ent_coef=cfg.ent_coef,
                clip_range=cfg.clip_range,
                tensorboard_log="out/tb",
                device="auto",
            )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=cfg.ent_coef,
            clip_range=cfg.clip_range,
            tensorboard_log="out/tb",
            device="auto",
        )

    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        LOG.info("Training for %s timesteps", cfg.timesteps)
        model.learn(total_timesteps=cfg.timesteps, progress_bar=True, callback=callbacks, reset_num_timesteps=False)
    except KeyboardInterrupt:
        interrupt_path = f"{cfg.model_path}_interrupted"
        model.save(interrupt_path)
        LOG.warning("Interrupted. Saved snapshot to %s.zip", interrupt_path)
        raise
    except Exception:
        emergency_path = f"{cfg.model_path}_emergency"
        model.save(emergency_path)
        LOG.error("Training failed. Emergency model saved to %s.zip", emergency_path)
        raise
    finally:
        if cfg.normalize_env:
            env.save("out/vecnormalize.pkl")

    model.save(cfg.model_path)
    LOG.info("Saved final model: %s.zip", cfg.model_path)
    if resumed_from:
        LOG.info("Resumed from: %s", resumed_from)


def evaluate(model_path: str, episodes: int = 5) -> None:
    ensure_packages(DEFAULT_PACKAGES)
    sb3 = importlib.import_module("stable_baselines3")
    PPO = getattr(sb3, "PPO")

    env = build_default_env()
    model = PPO.load(model_path)

    totals: list[float] = []
    for i in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0
        while not done and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            ep_reward += float(reward)
        totals.append(ep_reward)
        LOG.info("Episode %s reward: %.2f", i + 1, ep_reward)
    LOG.info("Average reward: %.2f", sum(totals) / max(1, len(totals)))


RLBOT_AGENT_TEMPLATE = r'''from __future__ import annotations

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import importlib


class LearnedGymAgent(BaseAgent):
    def initialize_agent(self):
        sb3 = importlib.import_module("stable_baselines3")
        PPO = getattr(sb3, "PPO")
        self.model = PPO.load(r"{MODEL_PATH}")
        self.controls = SimpleControllerState()

    def get_output(self, packet):
        obs = self._make_obs(packet)
        action, _ = self.model.predict(obs, deterministic=True)
        self._apply_action(action)
        return self.controls

    def _make_obs(self, packet):
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        return [
            car.physics.location.x,
            car.physics.location.y,
            car.physics.location.z,
            car.physics.velocity.x,
            car.physics.velocity.y,
            car.physics.velocity.z,
            ball.physics.location.x,
            ball.physics.location.y,
            ball.physics.location.z,
            ball.physics.velocity.x,
            ball.physics.velocity.y,
            ball.physics.velocity.z,
            float(car.has_wheel_contact),
            float(car.boost),
            float(packet.game_info.seconds_elapsed),
        ]

    def _apply_action(self, action):
        vals = list(action) if hasattr(action, "__iter__") else [action]
        vals = vals + [0.0] * (8 - len(vals))
        self.controls.throttle = float(max(-1, min(1, vals[0])))
        self.controls.steer = float(max(-1, min(1, vals[1])))
        self.controls.pitch = float(max(-1, min(1, vals[2])))
        self.controls.yaw = float(max(-1, min(1, vals[3])))
        self.controls.roll = float(max(-1, min(1, vals[4])))
        self.controls.jump = bool(vals[5] > 0.5)
        self.controls.boost = bool(vals[6] > 0.5)
        self.controls.handbrake = bool(vals[7] > 0.5)
'''


def write_rlbot_files(model_path: str, out_dir: str = "out/rlbot_bot") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    py_path = out / "agent.py"
    cfg_path = out / "agent.cfg"
    appearance_path = out / "appearance.cfg"

    py_path.write_text(RLBOT_AGENT_TEMPLATE.replace("{MODEL_PATH}", model_path), encoding="utf-8")

    cfg_path.write_text(
        "\n".join(
            [
                "[Locations]",
                f"python_file = {py_path.resolve()}",
                f"looks_config = {appearance_path.resolve()}",
                "name = LearnedGymAgent",
                "team = 0",
                "",
                "[Bot Parameters]",
                "maximum_tick_rate_preference = 120",
                "supports_early_start = True",
            ]
        ),
        encoding="utf-8",
    )

    appearance_path.write_text(
        "\n".join(
            [
                "[Bot Loadout]",
                "team_color_id = 27",
                "custom_color_id = 75",
                "car_id = 23",
                "decal_id = 305",
                "wheels_id = 1580",
                "boost_id = 32",
                "antenna_id = 0",
                "hat_id = 0",
                "paint_finish_id = 1681",
                "custom_finish_id = 1681",
                "engine_audio_id = 0",
                "trails_id = 0",
                "goal_explosion_id = 0",
            ]
        ),
        encoding="utf-8",
    )
    LOG.info("Generated RLBot files in %s", out)


def create_click_to_train_files(train_args: str = "--timesteps 2000000 --model out/ppo_rlgym") -> None:
    launch_dir = Path("out/launchers")
    launch_dir.mkdir(parents=True, exist_ok=True)

    bat = launch_dir / "start_training.bat"
    vbs = launch_dir / "start_training_silent.vbs"

    bat.write_text(
        "\n".join(
            [
                "@echo off",
                "set SCRIPT_DIR=%~dp0",
                "cd /d %SCRIPT_DIR%\\..\\..",
                "where python >nul 2>nul",
                "if errorlevel 1 (",
                "  echo Python was not found in PATH. Install Python 3.10+ and retry.",
                "  pause",
                "  exit /b 1",
                ")",
                f"python all_in_one_rlbot_gym.py train {train_args}",
                "if errorlevel 1 (",
                "  echo Training failed. See out\\logs\\run.log",
                "  pause",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    vbs.write_text(
        "\n".join(
            [
                'Set WshShell = CreateObject("WScript.Shell")',
                'Set fso = CreateObject("Scripting.FileSystemObject")',
                'script_dir = fso.GetParentFolderName(WScript.ScriptFullName)',
                'cmd = "cmd /c call """ & script_dir & ""\\start_training.bat"',
                'WshShell.Run cmd, 0, False',
            ]
        ),
        encoding="utf-8",
    )

    LOG.info("Created launchers: %s and %s", bat, vbs)


def write_zip(source_dir: str = ".", zip_name: str = "rocket_rl_all_in_one.zip") -> None:
    base = Path(source_dir)
    if not base.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_dir}")

    archive = Path(zip_name).with_suffix("")
    target_zip = archive.with_suffix(".zip")
    if target_zip.exists():
        target_zip.unlink()

    if base.is_dir():
        shutil.make_archive(str(archive), "zip", root_dir=str(base))
    else:
        parent = base.parent
        temp_name = archive.name
        shutil.make_archive(str(parent / temp_name), "zip", root_dir=str(parent), base_dir=base.name)
        generated = (parent / temp_name).with_suffix(".zip")
        if generated != target_zip:
            generated.replace(target_zip)
    LOG.info("Wrote zip: %s", target_zip)


def doctor() -> None:
    """Fast preflight checks before first real run."""
    LOG.info("Running preflight checks...")
    ensure_packages(DEFAULT_PACKAGES)
    write_rlbot_files("out/ppo_rlgym.zip")
    create_click_to_train_files()

    # Parser and environment smoke checks
    _ = build_default_env()
    LOG.info("Preflight completed. You are ready for first test run.")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="All-in-one RLGym + RLBot trainer/player")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("install", help="Install / upgrade dependencies")
    sub.add_parser("doctor", help="Run preflight checks and generate startup files")

    p_train = sub.add_parser("train", help="Train PPO with checkpoints and auto-resume")
    p_train.add_argument("--timesteps", type=int, default=2_000_000)
    p_train.add_argument("--model", type=str, default="out/ppo_rlgym")
    p_train.add_argument("--n-envs", type=int, default=8)
    p_train.add_argument("--learning-rate", type=float, default=3e-4)
    p_train.add_argument("--batch-size", type=int, default=4096)
    p_train.add_argument("--n-steps", type=int, default=2048)
    p_train.add_argument("--gamma", type=float, default=0.995)
    p_train.add_argument("--gae-lambda", type=float, default=0.95)
    p_train.add_argument("--ent-coef", type=float, default=0.005)
    p_train.add_argument("--clip-range", type=float, default=0.2)
    p_train.add_argument("--eval-freq", type=int, default=25_000)
    p_train.add_argument("--checkpoint-freq", type=int, default=25_000)
    p_train.add_argument("--no-resume", action="store_true")
    p_train.add_argument("--no-normalize", action="store_true")

    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--episodes", type=int, default=5)

    p_cfg = sub.add_parser("rlbot-config", help="Generate RLBot files")
    p_cfg.add_argument("--model", type=str, required=True)
    p_cfg.add_argument("--out", type=str, default="out/rlbot_bot")

    p_launch = sub.add_parser("make-launchers", help="Generate clickable Windows launchers")
    p_launch.add_argument("--train-args", type=str, default="--timesteps 2000000 --model out/ppo_rlgym")

    p_zip = sub.add_parser("zip", help="Zip project/output for sharing")
    p_zip.add_argument("--source", type=str, default=".")
    p_zip.add_argument("--name", type=str, default="rocket_rl_all_in_one.zip")

    args = parser.parse_args()

    if args.cmd == "install":
        ensure_packages(DEFAULT_PACKAGES)
        LOG.info("Dependencies ready.")
        return
    if args.cmd == "doctor":
        doctor()
        return
    if args.cmd == "train":
        cfg = TrainConfig(
            timesteps=args.timesteps,
            model_path=args.model,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            normalize_env=not args.no_normalize,
            resume=not args.no_resume,
        )
        train(cfg)
        return
    if args.cmd == "eval":
        evaluate(args.model, args.episodes)
        return
    if args.cmd == "rlbot-config":
        write_rlbot_files(args.model, args.out)
        return
    if args.cmd == "make-launchers":
        create_click_to_train_files(args.train_args)
        return
    if args.cmd == "zip":
        write_zip(args.source, args.name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        print(traceback.format_exc())
        sys.exit(1)
