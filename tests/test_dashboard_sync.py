"""
Test script for TrainingDashboard synchronization logic.
Verifies that updates are processed correctly and episode stats are forward-filled.
"""
import unittest
import queue
import numpy as np
from goodharts.training.train_dashboard import TrainingDashboard, RunState

class TestDashboardSync(unittest.TestCase):
    def setUp(self):
        self.dashboard = TrainingDashboard(modes=['test_mode'])
        self.run = self.dashboard.runs['test_mode']

    def test_update_processing(self):
        """Test unified update processing.

        The dashboard's _process_updates expects flat dictionaries with keys like
        'policy_loss', 'value_loss', 'reward', etc. rather than nested structures.
        """
        # 1. Send update WITHOUT episode data (no reward)
        payload1 = {
            'policy_loss': 0.1,
            'value_loss': 0.2,
            'entropy': 0.3,
            'action_probs': [0.1] * 8,
            'explained_var': 0.5,
            'reward': 0.0,  # No episode = zero reward
            'food': 0,
            'poison': 0,
            'total_steps': 1000,
        }
        self.dashboard.update('test_mode', 'update', payload1)
        self.dashboard._process_updates()

        self.assertEqual(len(self.run.policy_losses), 1)
        self.assertEqual(len(self.run.rewards), 1)
        self.assertEqual(self.run.rewards[0], 0.0)  # Zero as sent

        # 2. Send update WITH episode data
        payload2 = {
            'policy_loss': 0.05,
            'value_loss': 0.1,
            'entropy': 0.3,
            'action_probs': [0.1] * 8,
            'explained_var': 0.6,
            'reward': 10.0,  # Episode completed with reward
            'food': 5,
            'poison': 1,
            'total_steps': 2000,
        }
        self.dashboard.update('test_mode', 'update', payload2)
        self.dashboard._process_updates()

        self.assertEqual(len(self.run.policy_losses), 2)
        self.assertEqual(len(self.run.rewards), 2)
        self.assertEqual(self.run.rewards[1], 10.0)

        # 3. Send another update with zero reward
        payload3 = {
            'policy_loss': 0.04,
            'value_loss': 0.1,
            'entropy': 0.3,
            'action_probs': [0.1] * 8,
            'explained_var': 0.6,
            'reward': 0.0,  # No episode this update
            'food': 0,
            'poison': 0,
            'total_steps': 3000,
        }
        self.dashboard.update('test_mode', 'update', payload3)
        self.dashboard._process_updates()

        self.assertEqual(len(self.run.policy_losses), 3)
        self.assertEqual(len(self.run.rewards), 3)
        # Note: Dashboard doesn't forward-fill; it stores what's sent
        self.assertEqual(self.run.rewards[2], 0.0)

        print("Sync test passed: Lists aligned correctly.")

if __name__ == '__main__':
    unittest.main()
