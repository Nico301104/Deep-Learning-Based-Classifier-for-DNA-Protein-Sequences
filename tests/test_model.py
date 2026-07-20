"""Teste unitare pentru src/model.py: forward pass pe un batch dummy."""

from __future__ import annotations

import torch

from src import config
from src.model import build_model


class TestDNA_CNN:
    def test_forward_pass_output_shape(self):
        model = build_model()
        model.eval()

        batch_size = 8
        dummy_input = torch.randn(batch_size, 4, config.MAX_SEQ_LENGTH)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, config.CNN_CONFIG.num_classes)

    def test_forward_pass_is_finite(self):
        model = build_model()
        model.eval()

        dummy_input = torch.randn(4, 4, config.MAX_SEQ_LENGTH)
        with torch.no_grad():
            output = model(dummy_input)

        assert torch.isfinite(output).all()

    def test_handles_variable_batch_size(self):
        model = build_model()
        model.eval()

        for batch_size in (1, 2, 16):
            dummy_input = torch.randn(batch_size, 4, config.MAX_SEQ_LENGTH)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 1)

    def test_handles_shorter_sequence_length(self):
        # modelul trebuie sa functioneze si pe secvente mai scurte decat
        # MAX_SEQ_LENGTH implicit, datorita global average pooling-ului
        model = build_model()
        model.eval()

        dummy_input = torch.randn(3, 4, 40)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (3, 1)

    def test_model_has_trainable_parameters(self):
        model = build_model()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0
