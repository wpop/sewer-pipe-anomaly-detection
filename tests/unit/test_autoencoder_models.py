import torch

from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder
from sewer_anomaly.models.decoder import Decoder
from sewer_anomaly.models.encoder import Encoder


def test_encoder_output_shape() -> None:
    model = Encoder(latent_channels=128)
    images = torch.rand(2, 3, 256, 256)

    latent = model(images)

    assert latent.shape == (2, 128, 16, 16)


def test_decoder_output_shape() -> None:
    model = Decoder(latent_channels=128)
    latent = torch.rand(2, 128, 16, 16)

    reconstruction = model(latent)

    assert reconstruction.shape == (2, 3, 256, 256)


def test_autoencoder_preserves_input_shape() -> None:
    model = ConvolutionalAutoencoder(latent_channels=128)
    images = torch.rand(2, 3, 256, 256)

    reconstruction = model(images)

    assert reconstruction.shape == images.shape


def test_autoencoder_output_range_is_normalized() -> None:
    model = ConvolutionalAutoencoder(latent_channels=128)
    images = torch.rand(2, 3, 256, 256)

    reconstruction = model(images)

    assert torch.all(reconstruction >= 0.0)
    assert torch.all(reconstruction <= 1.0)
