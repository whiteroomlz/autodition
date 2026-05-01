from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import utils
from src.utils.setuptools import RequiresSetupABCMeta, RequiresSetupMeta, requires_setup

from .components.collate import Collator
from .components.dataset import (
    AudioDataset,
    FlatDataset,
    SequentialDataset,
    SourceSeparationDataset,
)
from .components.preprocessing.audio import AudioPreprocessingUnit, MelSpectrogram
from .components.preprocessing.sequential import Pipeline
from .components.preprocessing.sequential.augmentations import Augmentation
from .components.preprocessing.sequential.transforms import Transform
from .components.raw_data import DfData
from .components.schema import Schema

log = utils.RankedLogger(__name__, log_on_rank_zero_only=True)


class RawData(metaclass=RequiresSetupMeta):
    _feature_data: DfData = None
    _target_data: Optional[DfData] = None
    _train_keys: Optional[Union[DfData, Tuple[DfData]]] = None
    _val_keys: Optional[Union[DfData, Tuple[DfData]]] = None
    _test_keys: Optional[Union[DfData, Tuple[DfData]]] = None

    def __init__(
        self,
        feature_data_cfg: DictConfig,
        target_data_cfg: DictConfig,
        train_keys_cfg: DictConfig,
        val_keys_cfg: DictConfig,
        test_keys_cfg: DictConfig,
        **kwargs,
    ):
        self.feature_data_cfg = feature_data_cfg
        self.target_data_cfg = target_data_cfg
        self.train_keys_cfg = train_keys_cfg
        self.val_keys_cfg = val_keys_cfg
        self.test_keys_cfg = test_keys_cfg

    def setup(self):
        self._feature_data: DfData = hydra.utils.instantiate(self.feature_data_cfg)
        self._target_data: Optional[DfData] = hydra.utils.instantiate(self.target_data_cfg)
        self._train_keys: Optional[Union[DfData, Tuple[DfData]]] = hydra.utils.instantiate(
            self.train_keys_cfg
        )
        self._val_keys: Optional[Union[DfData, Tuple[DfData]]] = hydra.utils.instantiate(
            self.val_keys_cfg
        )
        self._test_keys: Optional[Union[DfData, Tuple[DfData]]] = hydra.utils.instantiate(
            self.test_keys_cfg
        )

    @requires_setup
    def get_feature_data(self):
        return self._feature_data

    @requires_setup
    def get_target_data(self):
        return self._target_data

    @requires_setup
    def get_train_keys(self):
        return self._train_keys

    @requires_setup
    def get_val_keys(self):
        return self._val_keys

    @requires_setup
    def get_test_keys(self):
        return self._test_keys


class DataModule(LightningDataModule, ABC, metaclass=RequiresSetupABCMeta):
    train_datasets: List[Dataset] = None
    val_datasets: List[Dataset] = None
    test_datasets: List[Dataset] = None

    feature_data: DfData = None
    target_data: Optional[DfData] = None
    train_keys: Optional[Union[DfData, Tuple[DfData]]] = None
    val_keys: Optional[Union[DfData, Tuple[DfData]]] = None
    test_keys: Optional[Union[DfData, Tuple[DfData]]] = None

    def __init__(
        self,
        schema: Schema,
        raw_data: RawData,
        collator: Collator,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 2,
        persistent_workers: bool = False,
        use_train_balance_sampler: bool = False,
        balance_on_field: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["schema", "raw_data", "collator"])

        self.schema = schema
        self.raw_data = raw_data
        self.collator = collator
        self.train_sampler = None

    def prepare_data(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def setup(self, stage: Optional[str] = None):
        log.info("Setup datasets...")

        self.raw_data.setup()
        self.feature_data = self.raw_data.get_feature_data()
        self.target_data = self.raw_data.get_target_data()
        self.train_keys = self.raw_data.get_train_keys()
        self.val_keys = self.raw_data.get_val_keys()
        self.test_keys = self.raw_data.get_test_keys()

        if isinstance(self.train_keys, DfData):
            self.train_keys = (self.train_keys,)
        if isinstance(self.val_keys, DfData):
            self.val_keys = (self.val_keys,)
        if isinstance(self.test_keys, DfData):
            self.test_keys = (self.test_keys,)

        self.train_datasets = [self._setup_dataset(keys, is_train=True) for keys in self.train_keys]
        self.val_datasets = [self._setup_dataset(keys, is_train=False) for keys in self.val_keys]
        self.test_datasets = [self._setup_dataset(keys, is_train=False) for keys in self.test_keys]

        if self.hparams.use_train_balance_sampler:
            self._setup_train_balance_sampler()

    def _setup_train_balance_sampler(self) -> None:
        if self.target_data is None:
            raise ValueError("Train balance sampling requires target_data")

        balance_field = self.hparams.balance_on_field
        if balance_field is None:
            categorical_supervision_fields = self.schema.categorical_supervision_field_names()
            if len(categorical_supervision_fields) != 1:
                raise ValueError(
                    "balance_on_field is required when schema does not expose exactly one "
                    "categorical supervision field"
                )
            balance_field = categorical_supervision_fields[0]

        if balance_field not in self.schema.categorical_supervision_field_names():
            raise ValueError(
                f"balance_on_field='{balance_field}' must reference a categorical supervision field"
            )

        targets = np.array(
            [
                self.target_data[self.train_keys[0][idx]["key"]][balance_field]
                for idx in range(len(self.train_keys[0]))
            ]
        )
        targets_counts = {target: np.sum(targets == target) for target in np.unique(targets)}
        log.info(f"Tgt balance: {targets_counts}")

        sampler_weights = torch.tensor([1.0 / targets_counts[target] for target in targets])
        self.train_sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(sampler_weights),
            replacement=True,
        )

    @abstractmethod
    def _setup_dataset(self, keys: DfData, is_train: bool) -> Dataset:
        raise NotImplementedError

    @requires_setup
    def train_dataloader(self):
        collate_fn = deepcopy(self.collator)
        dataloaders = [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=(self.train_sampler is None),
                collate_fn=collate_fn,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
                sampler=self.train_sampler,
            )
            for dataset in self.train_datasets
        ]
        return dataloaders if len(dataloaders) > 1 else dataloaders[0]

    @requires_setup
    def val_dataloader(self):
        collate_fn = deepcopy(self.collator)
        dataloaders = [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams.val_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=collate_fn,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
            )
            for dataset in self.val_datasets
        ]
        return dataloaders if len(dataloaders) > 1 else dataloaders[0]

    @requires_setup
    def test_dataloader(self):
        collate_fn = deepcopy(self.collator)
        dataloaders = [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams.test_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=collate_fn,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
            )
            for dataset in self.test_datasets
        ]
        return dataloaders if len(dataloaders) > 1 else dataloaders[0]

    def get_one_batch(self):
        def _first_loader(dataloader):
            return dataloader[0] if isinstance(dataloader, list) else dataloader

        try:
            batch = next(iter(_first_loader(self.train_dataloader())))
        except (IndexError, StopIteration):
            try:
                batch = next(iter(_first_loader(self.val_dataloader())))
            except (IndexError, StopIteration):
                batch = next(iter(_first_loader(self.test_dataloader())))

        return batch


class FlatDataModule(DataModule):
    def _setup_dataset(self, samples_keys: DfData, is_train: bool) -> Dataset:
        dataset = FlatDataset(
            feature_data=self.feature_data,
            schema=self.schema,
            samples_keys=samples_keys,
            target_data=self.target_data,
        )
        dataset.setup()
        return dataset


class SequentialDataModule(DataModule):
    def __init__(
        self,
        transforms_cfg: DictConfig,
        augmentations_cfg: DictConfig,
        sequential_features_key: str = "sequential_features",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequential_features_key = sequential_features_key
        self.transforms: Optional[Pipeline[Transform]] = hydra.utils.instantiate(transforms_cfg)
        self.augmentations: Optional[Pipeline[Augmentation]] = hydra.utils.instantiate(
            augmentations_cfg
        )

    def _setup_dataset(self, samples_keys: DfData, is_train: bool) -> Dataset:
        dataset = SequentialDataset(
            feature_data=self.feature_data,
            schema=self.schema,
            sequential_features_key=self.sequential_features_key,
            samples_keys=samples_keys,
            target_data=self.target_data,
            transforms=self.transforms,
            augmentations=self.augmentations if is_train else None,
        )
        dataset.setup()
        return dataset


class BaseAudioDataModule(DataModule):
    def __init__(
        self,
        audio_path_key: str = "audio_path",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_path_key = audio_path_key
        self.audio_root_dir = audio_root_dir
        self.target_sr = target_sr
        self.clip_duration_seconds = clip_duration_seconds


class AudioDataModule(BaseAudioDataModule):
    def __init__(
        self,
        mel_spectrogram_cfg: Optional[DictConfig],
        audio_path_key: str = "audio_path",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = None,
        waveform_field_name: Optional[str] = "waveform",
        spectrogram_field_name: Optional[str] = "mel_spectrogram",
        waveform_augmentations_cfg: Optional[DictConfig] = None,
        spectrogram_augmentations_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(
            audio_path_key=audio_path_key,
            audio_root_dir=audio_root_dir,
            target_sr=target_sr,
            clip_duration_seconds=clip_duration_seconds,
            **kwargs,
        )

        self.mel_spectrogram: Optional[MelSpectrogram] = (
            hydra.utils.instantiate(mel_spectrogram_cfg)
            if mel_spectrogram_cfg is not None
            else None
        )
        self.waveform_field_name = waveform_field_name
        self.spectrogram_field_name = spectrogram_field_name
        self.waveform_augmentations: Optional[AudioPreprocessingUnit] = (
            hydra.utils.instantiate(waveform_augmentations_cfg)
            if waveform_augmentations_cfg is not None
            else None
        )
        self.spectrogram_augmentations: Optional[AudioPreprocessingUnit] = (
            hydra.utils.instantiate(spectrogram_augmentations_cfg)
            if spectrogram_augmentations_cfg is not None
            else None
        )

    def _setup_dataset(self, samples_keys: DfData, is_train: bool) -> Dataset:
        dataset = AudioDataset(
            feature_data=self.feature_data,
            schema=self.schema,
            mel_spectrogram=self.mel_spectrogram,
            audio_path_key=self.audio_path_key,
            audio_root_dir=self.audio_root_dir,
            target_sr=self.target_sr,
            clip_duration_seconds=self.clip_duration_seconds,
            waveform_field_name=self.waveform_field_name,
            spectrogram_field_name=self.spectrogram_field_name,
            samples_keys=samples_keys,
            target_data=self.target_data,
            waveform_augmentations=self.waveform_augmentations if is_train else None,
            spectrogram_augmentations=self.spectrogram_augmentations if is_train else None,
        )
        dataset.setup()
        return dataset


class SourceSeparationDataModule(BaseAudioDataModule):
    def __init__(
        self,
        audio_path_key: str = "audio_path",
        source_audio_paths_key: str = "source_audio_paths",
        audio_root_dir: Optional[str] = None,
        target_sr: int = 16000,
        clip_duration_seconds: Optional[float] = 10.0,
        mixture_field_name: str = "mixture_audio",
        sources_field_name: str = "sources_audio",
        source_activity_field_name: Optional[str] = "source_activity",
        max_num_sources: int = 4,
        **kwargs,
    ):
        super().__init__(
            audio_path_key=audio_path_key,
            audio_root_dir=audio_root_dir,
            target_sr=target_sr,
            clip_duration_seconds=clip_duration_seconds,
            **kwargs,
        )
        self.source_audio_paths_key = source_audio_paths_key
        self.mixture_field_name = mixture_field_name
        self.sources_field_name = sources_field_name
        self.source_activity_field_name = source_activity_field_name
        self.max_num_sources = max_num_sources

    def _setup_dataset(self, samples_keys: DfData, is_train: bool) -> Dataset:
        dataset = SourceSeparationDataset(
            feature_data=self.feature_data,
            schema=self.schema,
            audio_path_key=self.audio_path_key,
            source_audio_paths_key=self.source_audio_paths_key,
            audio_root_dir=self.audio_root_dir,
            target_sr=self.target_sr,
            clip_duration_seconds=self.clip_duration_seconds,
            mixture_field_name=self.mixture_field_name,
            sources_field_name=self.sources_field_name,
            source_activity_field_name=self.source_activity_field_name,
            max_num_sources=self.max_num_sources,
            samples_keys=samples_keys,
            target_data=self.target_data,
        )
        dataset.setup()
        return dataset
