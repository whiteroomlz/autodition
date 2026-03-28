from typing import Any, Dict, Optional, Sequence, Union

import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from src.models.components.base import Model, ModelInput, ModelOutputForClassification


class ASTAudioClassifier(Model):
    """Full-model wrapper for Hugging Face AST classification models."""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        load_pretrained: bool = True,
        sample_rate: int = 16000,
        num_mel_bins: int = 128,
        max_length: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        attn_implementation: Optional[str] = "sdpa",
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.load_pretrained = load_pretrained
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.mean = mean
        self.std = std
        self.attn_implementation = attn_implementation
        self.model_config = dict(model_config or {})

        self.model: Optional[ASTForAudioClassification] = None
        self.feature_extractor: Optional[ASTFeatureExtractor] = None

    def setup(self) -> None:
        if self.model is not None and self.feature_extractor is not None:
            return

        self.model = self._build_model(self.model_config)
        self.feature_extractor = self._build_feature_extractor()

    def forward(
        self,
        model_input: Union[ModelInput, Dict[str, Any]],
    ) -> ModelOutputForClassification:
        model_input = self._coerce_model_input(model_input)
        if model_input.raw is None:
            raise ValueError("ASTAudioClassifier expects raw waveforms in model_input.raw")
        if not torch.is_tensor(model_input.raw) and any(waveform is None for waveform in model_input.raw):
            raise ValueError("ASTAudioClassifier expects raw waveforms in model_input.raw")

        model = self._get_model()
        feature_extractor = self._get_feature_extractor()
        raw_waveforms = self._normalize_raw_waveforms(model_input.raw)
        feature_extractor_output = feature_extractor(
            raw_waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        feature_extractor_output = {
            key: value.to(self._get_model_device()) if torch.is_tensor(value) else value
            for key, value in feature_extractor_output.items()
        }

        output = model(**feature_extractor_output)
        return ModelOutputForClassification(logits=output.logits)

    def _normalize_raw_waveforms(self, raw_waveforms: Union[torch.Tensor, Sequence[Any]]) -> list:
        if torch.is_tensor(raw_waveforms):
            if raw_waveforms.ndim == 1:
                raw_waveforms = raw_waveforms.unsqueeze(0)
            return [waveform.detach().float().cpu().numpy() for waveform in raw_waveforms]

        normalized_waveforms = []
        for waveform in raw_waveforms:
            if torch.is_tensor(waveform):
                normalized_waveforms.append(waveform.detach().float().cpu().numpy())
            else:
                normalized_waveforms.append(waveform)
        return normalized_waveforms

    def _build_model(self, model_config: Dict[str, Any]) -> ASTForAudioClassification:
        model_config = {key: value for key, value in dict(model_config).items() if value is not None}

        if self.load_pretrained:
            return ASTForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                attn_implementation=self.attn_implementation,
            )

        config = ASTConfig(
            num_labels=self.num_classes,
            max_length=self.max_length,
            num_mel_bins=self.num_mel_bins,
            **model_config,
        )
        return ASTForAudioClassification(config)

    def _build_feature_extractor(self) -> ASTFeatureExtractor:
        if self.load_pretrained:
            feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            feature_extractor.sampling_rate = self.sample_rate
            feature_extractor.num_mel_bins = self.num_mel_bins
            feature_extractor.max_length = self.max_length
            feature_extractor.mean = self.mean
            feature_extractor.std = self.std
            return feature_extractor

        return ASTFeatureExtractor(
            sampling_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            max_length=self.max_length,
            mean=self.mean,
            std=self.std,
            return_attention_mask=False,
        )

    def _get_model_device(self) -> torch.device:
        return next(self._get_model().parameters()).device

    def _get_model(self) -> ASTForAudioClassification:
        if self.model is None:
            raise RuntimeError("ASTAudioClassifier.setup() must be called before forward().")
        return self.model

    def _get_feature_extractor(self) -> ASTFeatureExtractor:
        if self.feature_extractor is None:
            raise RuntimeError("ASTAudioClassifier.setup() must be called before forward().")
        return self.feature_extractor
