"""Unit tests for ``tasks/common/observations.py``."""

import math

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

from loot_bench.tasks.common.observations import rot6d_from_matrix


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class TestRot6dFromMatrix:
    """The sign-free rotation observation."""

    def test_identity_and_known_rotation(self):
        eye = torch.eye(3)
        assert torch.allclose(
            rot6d_from_matrix(eye),
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        # 90 deg about z sends the local x axis to world +y and y to world -x.
        assert torch.allclose(
            rot6d_from_matrix(_rot_z(math.pi / 2)),
            torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),
            atol=1e-6,
        )

    def test_is_the_first_two_columns(self):
        mat = _rot_z(0.7) @ _rot_y(-0.3)
        out = rot6d_from_matrix(mat)
        assert torch.allclose(out[:3], mat[:, 0], atol=1e-6)
        assert torch.allclose(out[3:], mat[:, 1], atol=1e-6)

    def test_batch_shapes(self):
        assert rot6d_from_matrix(torch.eye(3).expand(5, 3, 3)).shape == (5, 6)
        assert rot6d_from_matrix(torch.eye(3).expand(4, 2, 3, 3)).shape == (4, 2, 6)

    def test_axes_stay_orthonormal(self):
        angles = [i * 6.0 / 19 for i in range(20)]
        mats = torch.stack([_rot_z(a) @ _rot_y(a / 2) for a in angles])
        out = rot6d_from_matrix(mats)
        col0, col1 = out[:, :3], out[:, 3:]
        assert torch.allclose(col0.norm(dim=-1), torch.ones(20), atol=1e-6)
        assert torch.allclose(col1.norm(dim=-1), torch.ones(20), atol=1e-6)
        assert torch.allclose((col0 * col1).sum(-1), torch.zeros(20), atol=1e-6)

    def test_continuous_where_the_quaternion_jumps(self):
        """The reason this term exists.

        ``quat_from_matrix`` picks its branch by an argmax over four nearly
        equal candidates, so it returns the quaternion or its negation
        depending on float noise. The DK1 grasp site sits exactly on that
        four-way tie, so consecutive steps of one smooth motion come back with
        opposite signs. The 6D form has no such freedom: one rotation, one
        encoding.
        """
        # A sweep through the tie region: all four |q| components near 0.5.
        angles = torch.linspace(0.0, 2.0 * math.pi, 400)
        mats = torch.stack([_rot_z(float(a)) @ _rot_y(math.pi / 2) for a in angles])

        quat = quat_from_matrix(mats)
        quat_jump = (quat[1:] - quat[:-1]).norm(dim=-1).max()
        # Show the defect is real, so this test fails loudly if mjlab ever
        # makes the conversion sign-stable and the workaround becomes dead.
        assert quat_jump > 1.0, "expected quat_from_matrix to flip sign on this sweep"

        out = rot6d_from_matrix(mats)
        step = (out[1:] - out[:-1]).norm(dim=-1).max()
        assert step < 0.05, f"6D rotation should stay continuous, saw {step}"
