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

CONTACT_THRESHOLD = 0.1
"""[N] force above which a step counts as touching the object.

Matches the constant ``qc/evaluation.py`` thresholds ``gripper_contact`` at, so
``object_contact_steps`` below and qc's own ``num_gripper_contacts`` measure
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
    gripper_contact: float | None = None
    """[N] the latest **unfiltered** finger-pad contact force, or ``None`` if
    the task has no ``grip_contact`` sensor. Not a running total: ``qc``
    thresholds it once per step to count grips, so it has to carry this step's
    reading. Counts the pads touching each other -- see
    :attr:`object_contact` for the number that answers "touched the object"."""
    object_contact: float | None = None
    """[N] the latest finger-pad contact force **against the scored object**,
    or ``None`` if the task has no ``object_contact`` sensor."""
    object_contact_steps: int = 0
    """Steps so far whose :attr:`object_contact` was above
    :data:`CONTACT_THRESHOLD`. Accumulated, unlike the two forces: ``qc``
    averages the final step of each episode, so a per-episode total is the only
    shape in which "how long did it hold the cube" survives into ``eval.csv``.
    """
    _rewards: list[float] = field(default_factory=list)

    def record(
        self,
        reward: float,
        gripper_contact: float | None = None,
        object_contact: float | None = None,
    ) -> None:
        """Fold one step's reward, and its two contact forces, in."""
        self.length += 1
        self.total_reward += reward
        self._rewards.append(reward)
        self.gripper_contact = gripper_contact
        self.object_contact = object_contact
        if object_contact is not None and object_contact > CONTACT_THRESHOLD:
            self.object_contact_steps += 1
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

        ``gripper_contact`` is published **flat**, and only flat. At ``d8a428f``
        ``qc/evaluation.py`` reads the nested ``info["proprio"]["gripper_contact"]``
        to decide the key exists, then thresholds the flat
        ``info["gripper_contact"]`` unconditionally -- so the nested form alone
        is a ``KeyError`` mid-evaluation, while the flat form is the one shape
        that is safe on both branches.

        A task with no ``grip_contact`` sensor leaves the key out entirely,
        which is safe too: ``qc`` tests for it before reading it, and drops to
        ``gripper_contact = None``. That is deliberate -- emitting 0.0 for a
        task that cannot measure contact would put a number in ``eval.csv``
        that looks like "never touched it" and means "never asked".

        ``object_contact`` and its two derived keys are the ones to read for
        "did the policy touch the object". ``gripper_contact`` is unfiltered
        and a closed gripper reports its own pads, so ``qc``'s
        ``num_gripper_contacts`` counts those too. ``object_contact_steps``
        and ``object_touched`` are per-episode totals, so they mean something
        after ``qc`` averages the final step over episodes: mean steps in
        contact, and the fraction of episodes with any contact at all.
        """
        info = {
            "success": float(self.success),
            "episode_length": float(self.length),
            "episode_return": float(self.total_reward),
        }
        if self.gripper_contact is not None:
            info["gripper_contact"] = self.gripper_contact
        if self.object_contact is not None:
            info["object_contact"] = self.object_contact
            info["object_contact_steps"] = float(self.object_contact_steps)
            info["object_touched"] = float(self.object_contact_steps > 0)
        return info
