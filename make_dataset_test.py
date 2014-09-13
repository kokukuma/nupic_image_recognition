#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The test of make dataset
"""

import os
import numpy

from pylearn2.gui import patch_viewer
from pylearn2.utils import serial, string_utils
from pylearn2.datasets import preprocessing
from data import cifar10

"""
get data
"""

# make output dir
output_dir = os.path.abspath('./data/pylearn2_test')
serial.mkdir( output_dir )

# input
train = cifar10.CIFAR10(which_set = 'train', gcn=55.)


"""
preprocess
"""

# 1.
preprocessor = preprocessing.ZCA()
train.apply_preprocessor(preprocessor=preprocessor , can_fit=True)


# 2. こっちで作成したpklをshow_exampleで見ると全て真っ黒になってしまった...
#    使っている中身も同じだし, show_exampleで表示されるmin/mean/maxも, どちらも同じになっているんだけど...
#    Pipelineを使うとこうなるんだろうか...
#pipeline = preprocessing.Pipeline()
# patch: 8x8
# pipeline.items.append(
#     preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000)
# )
# GCN: mean=0, var=1 ?
# pipeline.items.append(
#     preprocessing.GlobalContrastNormalization(scale=55.0, sqrt_bias=0.0, use_std=False)
# )
# ZCA whitening:
# pipeline.items.append(
#     preprocessing.ZCA()
# )
#train.apply_preprocessor(preprocessor=pipeline , can_fit=True)


"""
show train object
"""
print train
rows = 10
cols = 10

# 全部取ってくるためには, get_topological_viewかな.
examples = train.get_batch_topo(rows*cols)

# これが無いと...
#  ValueError: When rescale is set to False, pixel values must lie in [-1,1]. Got [-1.611357, 2.001721].
examples = train.adjust_for_viewer(examples)
#examples /= numpy.abs(examples).max()


# 画像出力
pv = patch_viewer.PatchViewer((rows, cols), examples.shape[1:3], is_color=True)
for i in xrange(rows*cols):
    pv.add_patch(examples[i, :, :, :], activation=0.0, rescale=False)
pv.show()



"""
save
"""
train.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir + '/test.pkl', train)
serial.save(output_dir + '/preprocessor.pkl', preprocessor)
