# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# from mmcv.runner import BaseModule
# 由于mmcv版本大更新，runner包已被删除，原本runner下的BaseModule移至新引擎mmengine的model目录下
from mmengine.model import BaseModule
from ..builder import BACKBONES

@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    """Vision Transformer."""

    def __init__(self, *args, **kwargs):
        super().__init__()
