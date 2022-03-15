# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS
# distill
ALGORITHMS = MODELS
MUTABLES = MODELS
DISTILLERS = MODELS
OPS = MODELS
PRUNERS = MODELS
QUANTIZERS = MODELS
ARCHITECTURES = MODELS
MUTATORS = MODELS

ATTENTION = Registry('attention', parent=MMCV_ATTENTION)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)

# distill
def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)


def build_mutator(cfg):
    """Build mutator."""
    return MUTATORS.build(cfg)


def build_distiller(cfg):
    """Build distiller."""
    return DISTILLERS.build(cfg)


def build_pruner(cfg):
    """Build pruner."""
    return PRUNERS.build(cfg)


def build_mutable(cfg):
    """Build mutable."""
    return MUTABLES.build(cfg)


def build_op(cfg):
    """Build op."""
    return OPS.build(cfg)

