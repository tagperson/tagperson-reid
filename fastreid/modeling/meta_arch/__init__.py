# encoding: utf-8
"""
anonymous
anonymous
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
