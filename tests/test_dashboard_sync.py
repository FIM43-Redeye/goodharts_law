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
        """Test unified update processing."""
        # 1. Send update WITHOUT episodes
        payload1 = {
            'ppo': (0.1, 0.2, 0.3, [0.1]*8, 0.5), # p_loss, v_loss, ent, probs, ev
            'episodes': None,
            'steps': 1000
        }
        self.dashboard.update('test_mode', 'update', payload1)
        self.dashboard._process_updates()
        
        self.assertEqual(len(self.run.policy_losses), 1)
        self.assertEqual(len(self.run.rewards), 1)
        self.assertEqual(self.run.rewards[0], 0.0) # Default/Zero filled
        
        # 2. Send update WITH episodes
        payload2 = {
            'ppo': (0.05, 0.1, 0.3, [0.1]*8, 0.6),
            'episodes': {'reward': 10.0, 'food': 5, 'poison': 1},
            'steps': 2000
        }
        self.dashboard.update('test_mode', 'update', payload2)
        self.dashboard._process_updates()
        
        self.assertEqual(len(self.run.policy_losses), 2)
        self.assertEqual(len(self.run.rewards), 2)
        self.assertEqual(self.run.rewards[1], 10.0)
        
        # 3. Send update WITHOUT episodes (Should forward fill 10.0)
        payload3 = {
            'ppo': (0.04, 0.1, 0.3, [0.1]*8, 0.6),
            'episodes': None,
            'steps': 3000
        }
        self.dashboard.update('test_mode', 'update', payload3)
        self.dashboard._process_updates()
        
        self.assertEqual(len(self.run.policy_losses), 3)
        self.assertEqual(len(self.run.rewards), 3)
        self.assertEqual(self.run.rewards[2], 10.0) # Forward filled!
        
        print("Sync test passed: Lists aligned and forward-filled.")

if __name__ == '__main__':
    unittest.main()
