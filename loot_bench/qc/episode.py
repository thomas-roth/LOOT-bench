"""Per-episode bookkeeping for the qc adapter.

Kept apart from the environment so it can be tested without a simulator. The
rules here are small but each one has a way of going quietly wrong, and a
quietly wrong success metric reads exactly like a policy that never learns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loot_bench.qc.transitions import SUCCESS_REWARD

CONTACT_THRESHOLD = 0.1
"""[N] force above which a step counts as touching the object.

``qc/evaluation.py`` imports this value to threshold ``pad_contact_force``,
so ``object_contact_steps`` below and qc's own ``pad_contact_runs`` measure
the same kind of step and can be read side by side. It equals
``tasks.common.commands.MIN_GRIP_FORCE`` as well, but for an unrelated reason
(that one is the physical release gate), so the two are not one number and
neither imports the other.
"""


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
    pad_contact_force: float | None = None
    """[N] the latest **unfiltered** finger-pad contact force, or ``None`` if
    the task has no ``grip_contact`` sensor. Not a running total: ``qc``
    thresholds it once per step to count grips, so it has to carry this step's
    reading."""
    object_contact_force: float | None = None
    """[N] the latest finger-pad contact force **against the scored object**,
    or ``None`` if the task has no ``object_contact`` sensor. Not a running
    total."""
    object_contact_steps: int = 0
    """Steps so far whose :attr:`object_contact_force` was above
    :data:`CONTACT_THRESHOLD`. Accumulated: ``qc`` averages the final step of
    each episode, so a per-episode total is the only shape in which "how long
    did it hold the cube" survives into ``eval.csv``.
    """
    _rewards: list[float] = field(default_factory=list)

    def record(
        self,
        reward: float,
        pad_contact_force: float | None = None,
        object_contact_force: float | None = None,
    ) -> None:
        """Fold one step's reward, and its two contact forces, in."""
        self.length += 1
        self.total_reward += reward
        self._rewards.append(reward)
        self.pad_contact_force = pad_contact_force
        self.object_contact_force = object_contact_force
        if (
            object_contact_force is not None
            and object_contact_force > CONTACT_THRESHOLD
        ):
            self.object_contact_steps += 1
        # Latched: the gate can unlatch after success, so the final step alone
        # is not enough.
        if reward >= SUCCESS_REWARD:
            self.success = True

    def info(self) -> dict[str, float]:
        """The per-step info dict.

        Published every step rather than only at the end: qc reads the last
        one, and a key that appears only sometimes is harder to trust than one
        that is always there.

        ``pad_contact_force`` is published **flat**, and only flat.
        ``qc/evaluation.py`` reads the flat key and falls back to the nested
        ``info["proprio"]`` form, so flat is the shape that is safe on both
        branches. The name is a cross-repo contract: rename it here and
        ``qc/evaluation.py`` stops matching, which is silent, the counters
        read 0.0 for every policy rather than raising.

        A task with no such sensor leaves the key out entirely, rather than
        emitting a 0.0 that reads as "never touched it" and means "never
        asked". ``qc`` tests for the key before reading it.

        ``object_touched`` is :attr:`object_contact_steps` ``> 0``, so once
        ``qc`` averages the final step over episodes it reads as the fraction
        of episodes with any contact at all.
        """
        info = {
            "success": float(self.success),
            "episode_length": float(self.length),
            "episode_return": float(self.total_reward),
        }

        if self.pad_contact_force is not None:
            info["pad_contact_force"] = self.pad_contact_force

        if self.object_contact_force is not None:
            info["object_contact_force"] = self.object_contact_force
            info["object_contact_steps"] = float(self.object_contact_steps)
            info["object_touched"] = float(self.object_contact_steps > 0)

        return info
