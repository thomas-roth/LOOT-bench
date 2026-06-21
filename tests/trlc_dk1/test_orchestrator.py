from unittest.mock import MagicMock

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.orchestrator import record_episode


class TestRecordEpisode:
    def test_records_one_episode(self):
        sim = MagicMock()
        sim.reset.return_value = {"actor": torch.zeros(1, 10)}
        sim.step.return_value = ({"actor": torch.zeros(1, 10)}, 0.0, False, {})
        recorder = MagicMock()
        actions = [np.zeros(7) for _ in range(5)]
        record_episode(sim, recorder, actions, task="test")
        assert sim.reset.called
        assert recorder.add_frame.call_count == 5
        recorder.end_episode.assert_called_once()

    def test_stops_on_done(self):
        sim = MagicMock()
        sim.reset.return_value = {"actor": torch.zeros(1, 10)}
        # Return done=True on third frame
        side_effects = [
            ({"actor": torch.zeros(1, 10)}, 0.0, False, {}),
            ({"actor": torch.zeros(1, 10)}, 0.0, False, {}),
            ({"actor": torch.zeros(1, 10)}, 1.0, True, {}),
        ]
        sim.step.side_effect = side_effects
        recorder = MagicMock()
        actions = [np.zeros(7) for _ in range(5)]
        record_episode(sim, recorder, actions, task="test")
        assert recorder.add_frame.call_count == 3
        recorder.end_episode.assert_called_once()
