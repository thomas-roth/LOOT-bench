from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .mdp import observations as obs
from .mdp.commands import MocapDeltaActionCfg
from .mdp.events import reset_peg_and_box
from .mdp.rewards import peg_insertion_reward
from .mdp.terminations import time_out

if TYPE_CHECKING:
    pass

_ASSETS = Path(__file__).parent / "assets"
_SAWYER_XML = _ASSETS / "sawyer_xyz" / "sawyer_peg_insertion_side.xml"
_PEG_XML = _ASSETS / "objects" / "peg.xml"
_BOX_XML = _ASSETS / "objects" / "box.xml"
_WORLD_XML = _ASSETS / "scene" / "world.xml"

# Box's authored 90 deg z-rotation (euler="0 0 1.57") as a (w, x, y, z) quat.
# mjlab overwrites the box root body's quat with init_state.rot at attach time,
# so the rotation must be supplied here, not only in the XML.
_BOX_ROT = (0.7073883, 0.0, 0.0, 0.7068252)


def _spec_from(path: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(path))


def _get_spec() -> mujoco.MjSpec:
    spec = _spec_from(_SAWYER_XML)
    # Make the robot base the entity's mocap root. The robot is fixed-base, so
    # mjlab would otherwise auto-wrap the whole entity in a "mocap_base" body to
    # position it per-env -- but that nests the robot's own control mocap body,
    # which must stay a direct child of world (MuJoCo rejects a mocap body that
    # is not a fixed child of world). Marking "base" mocap makes that auto-wrap
    # a no-op while still pinning the base per-env, leaving the control mocap at
    # world level. (Pre-split this was masked by the peg's freejoint, which made
    # the combined entity float and skipped the wrap.)
    spec.body("base").mocap = True
    return spec


_SAWYER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
)

_DEFAULT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={"r_close": 0.0, "l_close": 0.0},
    joint_vel={".*": 0.0},
)


def _get_sawyer_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_SAWYER_ARTICULATION,
        init_state=_DEFAULT_INIT,
    )


def _get_peg_cfg() -> EntityCfg:
    # Free-floating object: init_state.pos seeds the freejoint; the reset event
    # teleports it each episode.
    return EntityCfg(
        spec_fn=lambda: _spec_from(_PEG_XML),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.6, 0.03)),
    )


def _get_box_cfg() -> EntityCfg:
    # Static fixture: no joints. Pose is randomized at reset via model.body_pos;
    # init_state gives the default placement and the required z-rotation.
    return EntityCfg(
        spec_fn=lambda: _spec_from(_BOX_XML),
        init_state=EntityCfg.InitialStateCfg(pos=(-0.3, 0.6, 0.0), rot=_BOX_ROT),
    )


def _get_world_cfg() -> EntityCfg:
    # Static world (table, walls, floor, goal marker), single root at origin.
    return EntityCfg(
        spec_fn=lambda: _spec_from(_WORLD_XML),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def make_peg_insertion_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "eef_pos": ObservationTermCfg(
            func=obs.eef_pos_cartesian,
            params={"eef_cfg": obs._EEF_SITE_CFG},
        ),
        "gripper_open": ObservationTermCfg(
            func=obs.gripper_openness,
            params={"eef_cfg": obs._EEF_CFG},
        ),
        "peg_pos": ObservationTermCfg(
            func=obs.site_pos_abs,
            params={"asset_cfg": obs._PEG_GRASP_CFG},
        ),
        "eef_to_peg": ObservationTermCfg(
            func=obs.eef_to_peg_vec,
            params={"peg_cfg": obs._PEG_GRASP_CFG, "eef_cfg": obs._EEF_SITE_CFG},
        ),
        "peg_quat": ObservationTermCfg(
            func=obs.peg_quat,
        ),
        "eef_vel": ObservationTermCfg(
            func=obs.eef_vel_cartesian,
            params={"eef_cfg": obs._EEF_SITE_CFG},
        ),
        "goal_pos": ObservationTermCfg(
            func=obs.goal_pos_abs,
        ),
    }

    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "mocap_delta": MocapDeltaActionCfg(
            entity_name="sawyer",
            mocap_name="mocap",
            pos_scale=0.02,
            x_limits=(-0.5, 0.5),
            y_limits=(0.3, 0.9),
            z_limits=(0.0, 0.4),
        )
    }

    commands: dict[str, CommandTermCfg] = {}

    events = {
        "reset_peg_and_box": EventTermCfg(
            func=reset_peg_and_box,
            mode="reset",
            params={
                # randomize=False -> fixed, reproducible scene (peg in front of
                # the hole) for visual comparison against the native viewer.
                # Set True to sample peg/box from MetaWorld's ranges per reset
                # (domain randomization), which is what you want for RL training.
                "randomize": False,
            },
        ),
    }

    rewards = {
        "peg_insertion": RewardTermCfg(
            func=peg_insertion_reward,
            weight=1.0,
            params={
                "peg_cfg": obs._PEG_GRASP_CFG,
                "peg_head_cfg": obs._PEG_HEAD_CFG,
                "eef_cfg": obs._EEF_CFG,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    curriculum = {}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=None,
            entities={
                "world": _get_world_cfg(),
                "sawyer": _get_sawyer_cfg(),
                "box": _get_box_cfg(),
                "peg": _get_peg_cfg(),
            },
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="sawyer",
            body_name="base",
            distance=2.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                disableflags=("multiccd", "nativeccd"),
            ),
            nconmax=256,
            njmax=256,
        ),
        decimation=5,
        episode_length_s=50.0,
    )


def peg_insertion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_peg_insertion_env_cfg()
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
    return cfg


__all__ = [
    "make_peg_insertion_env_cfg",
    "peg_insertion_env_cfg",
    "_get_sawyer_cfg",
    "reset_peg_and_box",
]
