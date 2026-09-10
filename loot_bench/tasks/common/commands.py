"""Command-term helpers shared by the task categories.

The success metrics of the object-arrangement categories share a "pads are
force-free" release gate read off the finger-pad contact sensor
(:class:`ContactReleaseGate`).

The sensor's pad-geom match comes from the
:class:`~loot_bench.robots.interface.ManipulatorInterface` via
``manipulator_wiring.wire_manipulator_into_env``.
"""

from __future__ import annotations

import torch
from mjlab.sensor import ContactSensor

GRIP_CONTACT_SENSOR_NAME = "grip_contact"
"""Sensor name every category's release-gate contact sensor is registered
under (see ``manipulator_wiring.wire_manipulator_into_env``'s
``grip_contact_sensor`` param).

**Unfiltered on purpose**: it has no secondary match, so it reports the pads
touching *anything*. It does not report the pads touching each other: the
DK1's pads overlap slightly at full close, so ``dk1.xml`` excludes the pad
pair, and an empty gripper closed all the way reads zero here. A robot whose
pads can still touch needs the same exclusion, because the reading is used
unguarded elsewhere -- the stack command reads it to find the held block,
the proprioceptive ``pad_contact_force`` observation publishes it to the
policy, and ``qc/env.py`` publishes it as ``pad_contact_force`` telemetry.

That still leaves "touching *anything*": the number cannot answer "did it
touch the object". :data:`OBJECT_CONTACT_SENSOR_NAME` is the one that can.
"""

OBJECT_CONTACT_SENSOR_NAME = "object_contact"
"""Sensor name for the pads-against-the-manipulated-object contact sensor.

Same primary as ``grip_contact`` (the finger pads), plus a **secondary** match
on the object the task scores, so self-contact and table contact are filtered
out. Telemetry only: no success gate reads it. A category declares it by
adding the sensor and passing ``object_contact_sensor`` to
``registration.register_category``; a category with no single scored object
may leave it out, and the key then stays out of the info dict.
"""

# TODO: check with real robot
MIN_GRIP_FORCE = 0.1
"""[N] contact-force threshold above which a finger pad counts as still
gripping something."""


def pad_contact_force(sensor: ContactSensor) -> torch.Tensor:
    """(E,) strongest finger-pad contact force magnitude [N], per env.

    The one reduction of a pad contact sensor, used for both
    :data:`GRIP_CONTACT_SENSOR_NAME` and :data:`OBJECT_CONTACT_SENSOR_NAME`.
    :class:`ContactReleaseGate` thresholds it for the success gate, the
    proprioceptive ``pad_contact_force`` observation term publishes it to the
    policy, and ``qc/env.py`` publishes it as telemetry, so the number the
    policy sees, the number the episode ends on and a run's reported contact
    force all read the same value. What differs between the two sensors is
    the *filter*, never the reduction.
    """
    force = sensor.data.force
    assert force is not None
    return force.norm(dim=-1).amax(dim=-1)


class ContactReleaseGate:
    """Per-step "pads not exerting grip force" check for a command's
    success gate.

    This is the only release gate: it reads actual contact force at the
    finger pads, independent of joint position. A wedged or diagonal grasp
    can cross a joint-open threshold while still gripping, so joint position
    is not consulted. The sensor has no secondary (object) filter: it
    reports contact with anything, not just a tracked object, so this is
    coarser than "is it still holding the scored object".

    It does not include **pad-against-pad** contact: the DK1's pads overlap
    slightly at full close, so an empty gripper used to report grip force
    with nothing held, and ``dk1.xml`` excludes the pair. A robot whose pads
    can still touch needs the same exclusion, because the reading is used
    unguarded elsewhere -- the stack command reads it to find the held block,
    and ``qc/env.py`` publishes it as ``pad_contact_force`` telemetry.
    """

    def __init__(self, min_force: float = MIN_GRIP_FORCE):
        self._min_force = min_force

    def __call__(self, sensor: ContactSensor) -> torch.Tensor:
        return pad_contact_force(sensor) <= self._min_force
