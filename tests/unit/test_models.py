import unittest
import numpy as np
from satellite_srm.compat import torch
from satellite_srm.models.multispectral_swinir import MultispectralSwinIR
from satellite_srm.models.diffusion_sr import LightweightDiffusionSR
from satellite_srm.models.model_factory import create_model
from satellite_srm.training.trainer import SRMTrainer

class TestModels(unittest.TestCase):
    def test_swinir_forward_pass(self):
        model = MultispectralSwinIR(scale_factor=3.0, in_channels=4, out_channels=4)
        x = torch.from_numpy(np.random.rand(1, 4, 16, 16).astype(np.float32))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 48, 48))

    def test_diffusion_forward_pass(self):
        model = LightweightDiffusionSR(in_channels=4, out_channels=4, ddim_steps=2)
        x = torch.from_numpy(np.random.rand(1, 4, 16, 16).astype(np.float32))
        y = model(x, scale_factor=3.0)
        self.assertEqual(y.shape, (1, 4, 48, 48))

    def test_trainer_uses_configured_device(self):
        model = MultispectralSwinIR(scale_factor=2.0, in_channels=4, out_channels=4)
        dataset = [{"lr": np.random.rand(4, 8, 8).astype(np.float32), "hr": np.random.rand(4, 16, 16).astype(np.float32)}]
        trainer = SRMTrainer(model=model, train_dataset=dataset, val_dataset=dataset, config={"device": "cuda", "training": {"epochs": 1, "batch_size": 1}})

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(str(trainer.device), expected_device)
        self.assertEqual(next(model.parameters()).device.type, expected_device)

if __name__ == "__main__":
    unittest.main()
