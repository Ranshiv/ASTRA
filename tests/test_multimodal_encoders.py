"""multimodal_encoders.py: the four modality encoders and the physical-
scale-token fusion mechanism (backlog item 11)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import multimodal_encoders as enc

torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestSignedLogScale:
    def test_round_trips_positive_values(self):
        x = np.array([1.0, 50.0, 1e6])
        recovered = enc.inverse_signed_log_scale(enc.signed_log_scale(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_round_trips_negative_values(self):
        x = np.array([-1.0, -50.0])
        recovered = enc.inverse_signed_log_scale(enc.signed_log_scale(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_zero_maps_to_zero(self):
        assert enc.signed_log_scale(0.0) == pytest.approx(0.0)

    def test_compresses_large_dynamic_range(self):
        small = enc.signed_log_scale(np.array([1.0]))[0]
        large = enc.signed_log_scale(np.array([1e6]))[0]
        # A million-fold difference in raw scale becomes a small difference
        # in log space -- this is the whole point of the transform.
        assert large - small < 20


class TestResampleSpectrum:
    def test_output_shape(self):
        wave = np.linspace(4000, 7000, 50)
        flux = np.sin(np.linspace(0, 10, 50))
        error = np.full(50, 0.1)
        out = enc.resample_spectrum(wave, flux, error, length=32)
        assert out.shape == (3, 32)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            enc.resample_spectrum(np.array([4000.0]), np.array([1.0]), np.array([0.1]))

    def test_non_finite_points_are_dropped(self):
        wave = np.array([4000.0, 4100.0, np.nan, 4300.0, 4400.0])
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        error = np.full(5, 0.1)
        out = enc.resample_spectrum(wave, flux, error, length=8)
        assert np.all(np.isfinite(out))


class TestImageEncoder:
    def test_output_shape(self):
        model = enc.make_image_encoder(embedding_dim=16)
        x = torch.randn(4, 1, 16, 16)
        out = model.pooled(x)
        assert out.shape == (4, 16)

    def test_gradients_reach_every_parameter(self):
        model = enc.make_image_encoder(embedding_dim=8)
        x = torch.randn(2, 1, 16, 16)
        out = model.pooled(x)
        out.sum().backward()
        assert all(p.grad is not None for p in model.parameters())


class TestSpectrumEncoder:
    def test_output_shape(self):
        model = enc.make_spectrum_encoder(embedding_dim=16, length=64, patch_size=8)
        x = torch.randn(4, 3, 64)
        out = model.pooled(x)
        assert out.shape == (4, 16)

    def test_embedding_dim_not_divisible_by_heads_is_rejected(self):
        with pytest.raises(ValueError):
            enc.make_spectrum_encoder(embedding_dim=10, transformer_heads=4)


class TestCatalogEncoder:
    def test_output_shape(self):
        model = enc.make_catalog_encoder(embedding_dim=16, n_features=40)
        x = torch.randn(4, 40)
        out = model.pooled(x)
        assert out.shape == (4, 16)


class TestLightcurveEncoderReuse:
    def test_output_shape_matches_pretrain_encoder(self):
        model = enc.make_lightcurve_encoder(embedding_dim=16, length=64, patch_size=8)
        x = torch.randn(4, 2, 64)
        out = model.pooled(x)
        assert out.shape == (4, 16)

    def test_is_actually_pretrains_encoder_class(self):
        from astra import pretrain

        model = enc.make_lightcurve_encoder(embedding_dim=16, length=64, patch_size=8)
        reference = pretrain.make_encoder(
            pretrain.PretrainConfig(length=64, patch_size=8, transformer_dim=16))
        # `make_encoder` defines its `PatchEncoder` class inside its own
        # local scope, so each call produces a distinct class object even
        # with identical config -- compare by qualified name/architecture
        # instead of `type(...) is type(...)`.
        assert type(model).__qualname__ == type(reference).__qualname__
        assert isinstance(model.embed, torch.nn.Conv1d)
        assert model.embed.in_channels == reference.embed.in_channels == 2


class TestScaleTokenFusion:
    def test_scale_token_output_shape(self):
        token = enc.make_scale_token(embedding_dim=16)
        out = token(torch.randn(4, 1))
        assert out.shape == (4, 16)

    def test_fusion_output_shape(self):
        fusion = enc.make_scale_fusion(embedding_dim=16)
        out = fusion(torch.randn(4, 32))
        assert out.shape == (4, 16)

    def test_projection_head_output_shape(self):
        head = enc.make_projection_head(embedding_dim=16, projection_dim=8)
        out = head(torch.randn(4, 16))
        assert out.shape == (4, 8)

    def test_encode_and_fuse_shape_and_scale_sensitivity(self):
        model_modules = torch.nn.ModuleDict({
            "encoder": enc.make_catalog_encoder(embedding_dim=16, n_features=8),
            "scale_token": enc.make_scale_token(embedding_dim=16),
            "fusion": enc.make_scale_fusion(embedding_dim=16),
        })
        x = torch.randn(4, 8)
        low_scale = torch.zeros(4)
        high_scale = torch.full((4,), 10.0)

        fused_low = enc.encode_and_fuse(model_modules, x, low_scale)
        fused_high = enc.encode_and_fuse(model_modules, x, high_scale)

        assert fused_low.shape == (4, 16)
        # Different scale tokens for the same shape input must produce a
        # different fused embedding -- otherwise the scale token carries no
        # information at all, defeating the whole "brightness-aware" point.
        assert not torch.allclose(fused_low, fused_high)


class TestUnwiredFromProduction:
    def test_multimodal_modules_are_not_imported_by_rpc(self):
        from pathlib import Path

        rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
        assert "multimodal" not in rpc_source
