"""CartPole task environment configuration."""

import math

import torch

from mjlab.asset_zoo.robots.cartpole.cartpole_constants import get_cartpole_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig


def cartpole_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create CartPole environment configuration.

  Args:
    play: If True, disables corruption and extends episode length for evaluation.
  """

  # ==============================================================================
  # Scene Configuration
  # ==============================================================================

  scene_cfg = SceneCfg(
    num_envs=64 if not play else 16,  # Fewer envs for play mode
    extent=1.0,  # Spacing between environments
    entities={"robot": get_cartpole_robot_cfg()},
  )

  viewer_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="pole",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
  )

  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(
      timestep=0.02,  # 50 Hz control
      iterations=1,
    ),
  )

  # ==============================================================================
  # Actions
  # ==============================================================================

  actions = {
    "joint_pos": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=20.0,
      use_default_offset=False,
    ),
  }

  # ==============================================================================
  # Observations
  # ==============================================================================

  actor_terms = {
    "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2] / math.pi),
    "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2] / 5.0),
    "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1] / 2.0),
    "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1] / 20.0),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,  # Disable corruption in play mode
    ),
    "critic": ObservationGroupCfg(
      terms=actor_terms,  # Critic uses same observations
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ==============================================================================
  # Rewards
  # ==============================================================================

  def compute_upright_reward(env):
    """Reward for keeping pole upright (cosine of angle)."""
    return env.sim.data.qpos[:, 1].cos()

  def compute_effort_penalty(env):
    """Penalty for control effort."""
    return -0.01 * (env.sim.data.ctrl[:, 0] ** 2)

  rewards = {
    "upright": RewardTermCfg(func=compute_upright_reward, weight=5.0),
    "effort": RewardTermCfg(func=compute_effort_penalty, weight=1.0),
  }

  # ==============================================================================
  # Events
  # ==============================================================================

  def random_push_cart(env, env_ids, force_range=(-5, 5)):
    """Apply random force to cart for robustness training."""
    n = len(env_ids)
    random_forces = (
      torch.rand(n, device=env.device) * (force_range[1] - force_range[0])
      + force_range[0]
    )
    env.sim.data.qfrc_applied[env_ids, 0] = random_forces

  events = {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.1, 0.1),
      },
    ),
  }

  # Add random pushes only in training mode
  if not play:
    events["random_push"] = EventTermCfg(
      func=random_push_cart,
      mode="interval",
      interval_range_s=(1.0, 2.0),
      params={"force_range": (-20.0, 20.0)},
    )

  # ==============================================================================
  # Terminations
  # ==============================================================================

  def check_pole_tipped(env):
    """Check if pole has tipped beyond 30 degrees."""
    return env.sim.data.qpos[:, 1].abs() > math.radians(30)

  terminations = {
    "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "tipped": TerminationTermCfg(func=check_pole_tipped, time_out=False),
  }

  # ==============================================================================
  # Environment Configuration
  # ==============================================================================

  return ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    sim=sim_cfg,
    viewer=viewer_cfg,
    decimation=1,  # No action repeat
    episode_length_s=int(1e9) if play else 10.0,  # Infinite for play, 10s for training
  )
