import unittest
import torch
from src.config import SLMConfig, PipelineConfig
from src.device import DeviceManager, ExecutionMode
from src.model import PipelineStage, SequentialSLM
from src.pipeline import SequentialInferenceEngine, PipelineParallelEngine
from src.benchmark import generate_dummy_requests

class TestPipelineParallelSLM(unittest.TestCase):

    def setUp(self):
        self.slm_config = SLMConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=6,
            num_heads=2,
            intermediate_dim=128,
            max_seq_len=32
        )
        self.pipe_config = PipelineConfig(
            num_stages=2,
            micro_batch_size=2,
            total_requests=4,
            seq_len=16,
            force_cpu=True  # Ensure unit tests run deterministically on CPU
        )

    def test_device_manager_cpu(self):
        dev_mgr = DeviceManager(self.pipe_config)
        self.assertEqual(dev_mgr.mode, ExecutionMode.CPU)
        self.assertEqual(dev_mgr.get_stage_device(0), torch.device("cpu"))
        self.assertEqual(dev_mgr.get_stage_device(1), torch.device("cpu"))

    def test_stage_forward_shapes(self):
        dev_mgr = DeviceManager(self.pipe_config)
        stage0 = PipelineStage(0, self.slm_config, 0, 3)
        stage1 = PipelineStage(1, self.slm_config, 3, 6)

        dummy_tokens = torch.randint(0, self.slm_config.vocab_size, (2, 16))
        
        # Stage 0 forward
        activations = stage0(dummy_tokens)
        self.assertEqual(activations.shape, (2, 16, self.slm_config.hidden_dim))

        # Stage 1 forward
        logits = stage1(activations)
        self.assertEqual(logits.shape, (2, 16, self.slm_config.vocab_size))

    def test_sequential_vs_pipeline_equivalence(self):
        dev_mgr = DeviceManager(self.pipe_config)
        full_model = SequentialSLM(self.slm_config)

        seq_engine = SequentialInferenceEngine(self.pipe_config, self.slm_config, dev_mgr)
        pipe_engine = PipelineParallelEngine(self.pipe_config, self.slm_config, dev_mgr)

        seq_engine.set_weights(full_model)
        pipe_engine.set_weights(full_model)

        input_batches = generate_dummy_requests(
            total_requests=4,
            micro_batch_size=2,
            seq_len=16,
            vocab_size=self.slm_config.vocab_size
        )

        seq_outputs, seq_metrics = seq_engine.run(input_batches)
        pipe_outputs, pipe_metrics = pipe_engine.run(input_batches)

        self.assertEqual(len(seq_outputs), len(pipe_outputs))
        for seq_out, pipe_out in zip(seq_outputs, pipe_outputs):
            self.assertTrue(torch.allclose(seq_out, pipe_out, atol=1e-4))

    def test_pipeline_metrics(self):
        dev_mgr = DeviceManager(self.pipe_config)
        pipe_engine = PipelineParallelEngine(self.pipe_config, self.slm_config, dev_mgr)
        input_batches = generate_dummy_requests(4, 2, 16, 100)

        _, metrics = pipe_engine.run(input_batches)
        self.assertGreater(metrics.total_time_ms, 0.0)
        self.assertGreaterEqual(metrics.bubble_ratio, 0.0)
        self.assertLessEqual(metrics.bubble_ratio, 1.0)
        self.assertEqual(len(metrics.timeline_events), 4)  # 2 micro-batches * 2 stages = 4 events

if __name__ == "__main__":
    unittest.main()
