"""Core field-native model runtime built around refs, stages, and tensor slots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.components.batch import Batch
from src.utils.setuptools import (
    SETUP_FUNCTION_NAME,
    RequiresSetupABCMeta,
    requires_setup,
)

RefSource = Literal["field", "mask", "rep", "pred"]
StoreName = Literal["rep", "pred"]

REF_SOURCES = frozenset({"field", "mask", "rep", "pred"})
STORE_NAMES = frozenset({"rep", "pred"})


@dataclass(frozen=True)
class Ref:
    """Pointer to a tensor source in the runtime: batch field, mask, rep, or pred."""

    source: RefSource
    name: str

    def __post_init__(self) -> None:
        if self.source not in REF_SOURCES:
            raise ValueError(f"Unsupported ref source: {self.source}")
        if not self.name:
            raise ValueError("Ref.name must not be empty")


@dataclass
class TensorSlot:
    """Tensor payload optionally paired with a boolean mask."""

    value: torch.Tensor
    mask: Optional[torch.BoolTensor] = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.value):
            raise TypeError("TensorSlot.value must be a torch.Tensor")
        if self.mask is not None:
            if not torch.is_tensor(self.mask):
                raise TypeError("TensorSlot.mask must be a torch.BoolTensor")
            self.mask = self.mask.bool()


@dataclass
class ModelContext:
    """Mutable execution state while a pipeline consumes one collated batch."""

    batch: Batch
    reps: Dict[str, TensorSlot] = field(default_factory=dict)
    preds: Dict[str, TensorSlot] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch: Batch) -> ModelContext:
        return cls(batch=batch, meta=dict(batch.meta))

    def resolve_slot(self, ref: Ref) -> TensorSlot:
        if ref.source == "field":
            if ref.name not in self.batch.fields:
                raise KeyError(f"Batch does not contain field '{ref.name}'")
            value = self.batch.fields[ref.name]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Field '{ref.name}' must be tensor-collated before crossing model boundary"
                )
            return TensorSlot(value=value, mask=self.batch.masks.get(ref.name))

        if ref.source == "mask":
            if ref.name not in self.batch.masks:
                raise KeyError(f"Batch does not contain mask '{ref.name}'")
            return TensorSlot(value=self.batch.masks[ref.name].bool())

        storage = self.reps if ref.source == "rep" else self.preds
        if ref.name not in storage:
            raise KeyError(f"{ref.source} '{ref.name}' is not available in the current context")
        return storage[ref.name]

    def resolve_tensor(self, ref: Ref) -> torch.Tensor:
        return self.resolve_slot(ref).value

    def resolve_mask(self, ref: Ref) -> torch.BoolTensor:
        if ref.source == "mask":
            return self.resolve_tensor(ref).bool()

        slot = self.resolve_slot(ref)
        if slot.mask is None:
            raise KeyError(f"Ref '{ref.source}:{ref.name}' does not expose a mask")
        return slot.mask.bool()

    def write(self, store: StoreName, name: str, slot: TensorSlot) -> None:
        if store not in STORE_NAMES:
            raise ValueError(f"Unsupported store '{store}'")

        storage = self.reps if store == "rep" else self.preds
        if name in storage:
            raise ValueError(f"{store} '{name}' already has a producer")
        storage[name] = slot


@dataclass
class ModelResult:
    """Immutable model outputs exposed to objectives, metrics, and callers."""

    reps: Dict[str, TensorSlot] = field(default_factory=dict)
    preds: Dict[str, TensorSlot] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: ModelContext) -> ModelResult:
        return cls(
            reps=dict(context.reps),
            preds=dict(context.preds),
            meta=dict(context.meta),
        )

    def resolve_slot(self, ref: Ref) -> TensorSlot:
        if ref.source == "rep":
            if ref.name not in self.reps:
                raise KeyError(f"Result does not contain rep '{ref.name}'")
            return self.reps[ref.name]
        if ref.source == "pred":
            if ref.name not in self.preds:
                raise KeyError(f"Result does not contain pred '{ref.name}'")
            return self.preds[ref.name]
        raise KeyError(f"ModelResult cannot resolve non-model ref '{ref.source}:{ref.name}'")


class Stage(torch.nn.Module, ABC):
    """Ordered pipeline unit that reads refs and writes named reps or preds."""

    def __init__(self, inputs: Sequence[Ref], outputs: Sequence[str], store: StoreName) -> None:
        super().__init__()
        self.inputs = tuple(_coerce_ref(ref) for ref in inputs)
        self.outputs = tuple(outputs)
        self.store = store

        if store not in STORE_NAMES:
            raise ValueError(f"Unsupported stage store '{store}'")
        if not self.outputs:
            raise ValueError("Stage must declare at least one output")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Stage outputs must be unique")

    @abstractmethod
    def forward(self, context: ModelContext) -> ModelContext:
        raise NotImplementedError


class EncoderStage(Stage, ABC):
    """Stage family for producing intermediate representations from batch inputs."""

    def __init__(self, inputs: Sequence[Ref], outputs: Sequence[str]) -> None:
        super().__init__(inputs=inputs, outputs=outputs, store="rep")


class TransformStage(Stage, ABC):
    """Stage family for transforming existing reps or preds into new reps."""

    def __init__(self, inputs: Sequence[Ref], outputs: Sequence[str]) -> None:
        super().__init__(inputs=inputs, outputs=outputs, store="rep")


class HeadStage(Stage, ABC):
    """Stage family for producing task-facing predictions."""

    def __init__(self, inputs: Sequence[Ref], outputs: Sequence[str]) -> None:
        super().__init__(inputs=inputs, outputs=outputs, store="pred")


class InputBlock(torch.nn.Module, ABC):
    """First block inside a BlockModel, consuming one or more resolved input slots."""

    @abstractmethod
    def forward(self, inputs: Sequence[TensorSlot], context: ModelContext) -> TensorSlot:
        raise NotImplementedError


class HiddenBlock(torch.nn.Module, ABC):
    """Intermediate block operating on a single slot inside a BlockModel."""

    @abstractmethod
    def forward(self, slot: TensorSlot, context: ModelContext) -> TensorSlot:
        raise NotImplementedError


class OutputBlock(torch.nn.Module, ABC):
    """Final block inside a BlockModel, usually producing task-ready tensors."""

    @abstractmethod
    def forward(self, slot: TensorSlot, context: ModelContext) -> TensorSlot:
        raise NotImplementedError


class Model(torch.nn.Module, ABC, metaclass=RequiresSetupABCMeta):
    """Top-level model contract: batch in, model result out, setup-aware."""

    def setup(self) -> None:
        """Prepare model resources after Hydra instantiation."""

    @abstractmethod
    @requires_setup
    def forward(self, batch: Batch) -> ModelResult:
        raise NotImplementedError


class BlockModel(Stage):
    """Linear stage adapter reusing input, hidden, and output blocks."""

    def __init__(
        self,
        inputs: Sequence[Ref],
        output_name: str,
        store: StoreName,
        input_block: InputBlock,
        hidden_blocks: Sequence[HiddenBlock],
        output_block: OutputBlock,
    ) -> None:
        super().__init__(inputs=inputs, outputs=(output_name,), store=store)
        self.input_block = input_block
        self.hidden_blocks = torch.nn.ModuleList(hidden_blocks)
        self.output_block = output_block

    def forward(self, context: ModelContext) -> ModelContext:
        input_slots = tuple(context.resolve_slot(ref) for ref in self.inputs)
        slot = self.input_block(input_slots, context)
        for block in self.hidden_blocks:
            slot = block(slot, context)
        slot = self.output_block(slot, context)
        context.write(self.store, self.outputs[0], slot)
        return context


class PipelineModel(Model):
    """Explicitly ordered stage pipeline without graph auto-topology magic."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        super().__init__()
        self.stages = torch.nn.ModuleList(stages)

    def setup(self) -> None:
        self._validate_stage_wiring()

    def _validate_stage_wiring(self) -> None:
        available_reps = set()
        available_preds = set()

        for stage in self.stages:
            for ref in stage.inputs:
                if ref.source == "rep" and ref.name not in available_reps:
                    raise ValueError(
                        f"Stage '{stage.__class__.__name__}' depends on unavailable rep '{ref.name}'"
                    )
                if ref.source == "pred" and ref.name not in available_preds:
                    raise ValueError(
                        f"Stage '{stage.__class__.__name__}' depends on unavailable pred '{ref.name}'"
                    )

            target_storage = available_reps if stage.store == "rep" else available_preds
            for output_name in stage.outputs:
                if output_name in target_storage:
                    raise ValueError(f"{stage.store} '{output_name}' already has a producer")
                target_storage.add(output_name)

    def forward(self, batch: Batch) -> ModelResult:
        context = ModelContext.from_batch(batch)
        for stage in self.stages:
            context = stage(context)
        return ModelResult.from_context(context)


class BackboneWithHeads(Model):
    """Small convenience wrapper for the common backbone-plus-heads layout."""

    def __init__(
        self,
        backbone: Sequence[Stage],
        heads: Sequence[Stage],
        transforms: Optional[Sequence[Stage]] = None,
    ) -> None:
        super().__init__()
        self.pipeline = PipelineModel(stages=[*backbone, *(transforms or ()), *heads])

    def setup(self) -> None:
        self.pipeline.setup()

    def forward(self, batch: Batch) -> ModelResult:
        return self.pipeline(batch)


def setup_modules(module: torch.nn.Module) -> None:
    """Recursively call setup on a module tree where components expose it."""

    setup_method = getattr(module, SETUP_FUNCTION_NAME, None)
    if callable(setup_method):
        setup_method()

    for child_module in module.children():
        setup_modules(child_module)


def _coerce_ref(ref: Ref | Dict[str, str]) -> Ref:
    if isinstance(ref, DictConfig):
        ref = OmegaConf.to_object(ref)
    if isinstance(ref, Ref):
        return ref
    if isinstance(ref, dict):
        ref_dict = {key: value for key, value in ref.items() if not key.startswith("_")}
        return Ref(**ref_dict)
    raise TypeError(f"Unsupported ref type: {type(ref).__name__}")


def ensure_single_input(inputs: Sequence[TensorSlot], block_name: str) -> TensorSlot:
    if len(inputs) != 1:
        raise ValueError(f"{block_name} expects exactly one input slot")
    return inputs[0]


def resolve_refs(context: ModelContext, refs: Iterable[Ref]) -> Tuple[TensorSlot, ...]:
    return tuple(context.resolve_slot(ref) for ref in refs)
