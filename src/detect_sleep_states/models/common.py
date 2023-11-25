from typing import Union, Callable

import torch
import torch.nn as nn

from detect_sleep_states.config import DecoderConfig, FeatureExtractorConfig, InferenceConfig, TrainConfig
from detect_sleep_states.models.base import BaseModel
from detect_sleep_states.models.centernet import CenterNet
from detect_sleep_states.models.decoder.lstmdecoder import LSTMDecoder
from detect_sleep_states.models.decoder.mlpdecoder import MLPDecoder
from detect_sleep_states.models.decoder.transformercnndecoder import TransformerCNNDecoder
from detect_sleep_states.models.decoder.transformerdecoder import TransformerDecoder
from detect_sleep_states.models.decoder.unet1ddecoder import UNet1DDecoder
from detect_sleep_states.models.detr2D import DETR2DCNN
from detect_sleep_states.models.feature_extractor.cnn import CNNSpectrogram
from detect_sleep_states.models.feature_extractor.lstm import LSTMFeatureExtractor
from detect_sleep_states.models.feature_extractor.panns import PANNsFeatureExtractor
from detect_sleep_states.models.feature_extractor.spectrogram import SpecFeatureExtractor
from detect_sleep_states.models.spec1D import Spec1D
from detect_sleep_states.models.spec2Dcnn import Spec2DCNN

FEATURE_EXTRACTOR_TYPE = Union[
    CNNSpectrogram, PANNsFeatureExtractor, LSTMFeatureExtractor, SpecFeatureExtractor
]
DECODER_TYPE = Union[
    UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder, TransformerCNNDecoder
]


def get_feature_extractor(
        cfg: FeatureExtractorConfig, feature_dim: int, num_timesteps: int
) -> FEATURE_EXTRACTOR_TYPE:
    feature_extractor: FEATURE_EXTRACTOR_TYPE
    if cfg.name == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim, output_size=num_timesteps, **cfg.params
        )
    elif cfg.name == "PANNsFeatureExtractor":
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim, output_size=num_timesteps, conv=nn.Conv1d, **cfg.params
        )
    elif cfg.name == "LSTMFeatureExtractor":
        feature_extractor = LSTMFeatureExtractor(
            in_channels=feature_dim, out_size=num_timesteps, **cfg.params
        )
    elif cfg.name == "SpecFeatureExtractor":
        feature_extractor = SpecFeatureExtractor(
            in_channels=feature_dim, out_size=num_timesteps, **cfg.params
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.name}")

    return feature_extractor


def get_decoder(
        cfg: DecoderConfig, n_channels: int, n_classes: int, num_timesteps: int
) -> DECODER_TYPE:
    decoder: DECODER_TYPE
    if cfg.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            **cfg.params,
        )
    elif cfg.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    elif cfg.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    elif cfg.name == "MLPDecoder":
        decoder = MLPDecoder(n_channels=n_channels, n_classes=n_classes)
    elif cfg.name == "TransformerCNNDecoder":
        decoder = TransformerCNNDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg.name}")

    return decoder


def get_model(
        cfg: TrainConfig | InferenceConfig,
        feature_dim: int,
        n_classes: int,
        num_timesteps: int,
        test: bool = False,
) -> BaseModel:
    model: BaseModel
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(cfg.decoder, feature_extractor.height, n_classes, num_timesteps)
        if test:
            cfg.model.params["encoder_weights"] = None
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "Spec1D":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        n_channels = feature_extractor.out_chans * feature_extractor.height
        decoder = get_decoder(cfg.decoder, n_channels, n_classes, num_timesteps)
        model = Spec1D(
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
        )
    elif cfg.model.name == "DETR2DCNN":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(
            cfg.decoder, feature_extractor.height, cfg.model.params["hidden_dim"], num_timesteps
        )
        if test:
            cfg.model.params["encoder_weights"] = None
        model = DETR2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "CenterNet":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(cfg.decoder, feature_extractor.height, 6, num_timesteps)
        if test:
            cfg.model.params["encoder_weights"] = None
        model = CenterNet(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")

    return model
