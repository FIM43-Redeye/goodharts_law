
import unittest
import torch
import os
import shutil
import tempfile
from goodharts.training.ppo.trainer import PPOTrainer, PPOConfig

class TestPPODeepSanity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_deep_model.pth")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_hyper_verbose_learning(self):
        """Run PPO with hyper_verbose=True to trigger debug prints."""
        # Config for minimal run but enough steps to get stats
        n_envs = 16
        steps_per_env = 32 # Enough for GAE to have some horizon
        
        config = PPOConfig(
            mode='ground_truth',
            brain_type='base_cnn',
            n_envs=n_envs,
            total_timesteps=n_envs * steps_per_env * 2, # 2 updates
            steps_per_env=steps_per_env,
            n_minibatches=4,
            output_path=self.model_path,
            log_to_file=False,
            use_amp=False,
            compile_models=False,
            compile_env=False,  # Both compile flags off for fast tests
            skip_warmup=True,
            use_torch_env=True,
            hyper_verbose=True # <--- TRIGGERS DEBUG PRINTS
        )
        
        print(f"\n[DeepSanity] Starting hyper-verbose training run...")
        trainer = PPOTrainer(config)
        trainer.train()
        print(f"\n[DeepSanity] Run complete.")

if __name__ == '__main__':
    unittest.main()
