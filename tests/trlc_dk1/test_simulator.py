from unittest.mock import MagicMock, patch

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.simulator import Simulator


class TestSimulator:
    @patch("g3a_bench.scripts.live_bridge.simulator.ManagerBasedRlEnv")
    def test_creates_env(self, mock_env_cls):
        Simulator(env_cfg=MagicMock())
        mock_env_cls.assert_called_once()

    @patch("g3a_bench.scripts.live_bridge.simulator.ManagerBasedRlEnv")
    def test_reset_returns_obs(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"actor": torch.zeros(1, 10)}, {})
        mock_env_cls.return_value = mock_env
        sim = Simulator(env_cfg=MagicMock())
        obs = sim.reset()
        assert "actor" in obs
        mock_env.reset.assert_called_once()

    @patch("g3a_bench.scripts.live_bridge.simulator.ManagerBasedRlEnv")
    def test_step_returns_obs_reward_done(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.device = "cpu"
        mock_env.step.return_value = (
            {"actor": torch.zeros(1, 10)},
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )
        mock_env_cls.return_value = mock_env
        sim = Simulator(env_cfg=MagicMock())
        obs, reward, done, extras = sim.step(np.zeros(7))
        assert reward == 1.0
        assert done is False

    @patch("g3a_bench.scripts.live_bridge.simulator.ManagerBasedRlEnv")
    def test_step_1d_to_2d(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env.device = "cpu"
        mock_env.step.return_value = (
            {"actor": torch.zeros(1, 10)},
            torch.tensor([0.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            {},
        )
        mock_env_cls.return_value = mock_env
        sim = Simulator(env_cfg=MagicMock())
        _obs, reward, done, _extras = sim.step(np.zeros(7))
        assert done is True
        called_action = mock_env.step.call_args[0][0]
        assert called_action.shape == (1, 7)

    @patch("g3a_bench.scripts.live_bridge.simulator.ManagerBasedRlEnv")
    def test_close_delegates(self, mock_env_cls):
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        sim = Simulator(env_cfg=MagicMock())
        sim.close()
        mock_env.close.assert_called_once()
