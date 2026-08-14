"""Per-episode bookkeeping for the qc adapter.

Kept apart from the environment so it can be tested without a simulator. The
rules here are small but each one has a way of going quietly wrong, and a
quietly wrong success metric reads exactly like a policy that never learns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The reward convention (tasks/common/rewards.py) puts success at exactly 0.0
# and everything else at or below -MIN_SHORTFALL. So a reward of 0.0 *is* the
# success signal, and the adapter needs no second opinion.
SUCCESS_REWARD = 0.0


@dataclass
class EpisodeStats:
    """Running totals for one episode, and the info dict qc reads at the end.

    ``qc/evaluation.py`` builds its statistics from ``flatten(info)`` on the
    **final** info dict of each episode, and averages the result over episodes.
    So every number that should appear in ``eval/`` has to be present on the
    last step, already reduced to a scalar.
    """

    length: int = 0
    total_reward: float = 0.0
    success: bool = False
    _rewards: list[float] = field(default_factory=list)

    def record(self, reward: float) -> None:
        """Fold one step's reward in."""
        self.length += 1
        self.total_reward += reward
        self._rewards.append(reward)
        # Latched, not sampled. A task's success gate can unlatch -- a block
        # settles a millimetre and the at-rest test drops out -- and the
        # episode ends a step later. Reading success off the final step alone
        # would then score a successful episode as a failure.
        if reward >= SUCCESS_REWARD:
            self.success = True

    def info(self) -> dict[str, float]:
        """The per-step info dict.

        Published every step rather than only at the end: qc reads the last
        one, and a key that appears only sometimes is harder to trust than one
        that is always there.

        Deliberately does **not** publish ``gripper_contact``. At ``d8a428f``
        ``qc/evaluation.py`` reads the nested ``info["proprio"]["gripper_contact"]``
        to decide the key exists, then thresholds the flat
        ``info["gripper_contact"]`` unconditionally -- so publishing only the
        nested form is a ``KeyError`` mid-evaluation. It is optional telemetry,
        so the safe move is not to emit it.
        """
        return {
            "success": float(self.success),
            "episode_length": float(self.length),
            "episode_return": float(self.total_reward),
        }
