#!/usr/bin/python
# coding: utf-8

# 1. pylearn2のdatasetを読み込み.
# 2. 指定した形状でdataを取得

from pylearn2.utils import serial
import numpy

def load_dataset(path):
    """
    pylearn2 dataset
    """
    # load pylearn2 dataset
    obj = serial.load(path)
    obj.set_view_converter_axes(['b', 'c', 0, 1])

    image_data, label = obj.get_data()
    tobological_data  = obj.get_topological_view(image_data)

    # convert ['b', 'c', 0, 1] -> ['b', 0, 1, 'c']
    tobological_data = numpy.transpose(tobological_data, (0,2,3,1))
    return tobological_data, label


def get_patch(tdata, height=3, width=3, step=1, type='slide'):
    """
    tdata : ['b', 0, 1, 'c']
            全部のデータを分割してしまおうかと思ったが, デカ過ぎで難しい...
            cla networkに入れるとき分割することにする.
    """
    patch_x = tdata.shape[0] - width  + 1
    patch_y = tdata.shape[1] - height + 1
    patch_len = len(range(0, patch_x, step)) * len(range(0, patch_y, step))

    patchs    = numpy.zeros([ patch_len, height, width, tdata.shape[2]], dtype='float')
    movement  = numpy.zeros([ patch_len, 2], dtype='float')

    def get_patch_data(data, base_x, base_y, patch_id):
        for y in range(height):
            for x in range(width):
                patchs[patch_id][y][x] = data[base_y+y][base_x+x]
        return

    # explorer
    patch_id = 0
    for j, y_id in enumerate(range(0, patch_y, step)):
        rev = (j % 2 == 1)
        for x_id in sorted(range(0, patch_x, step), reverse=rev):
            print patch_id,x_id, y_id
            get_patch_data(tdata, x_id, y_id, patch_id)
            movement[patch_id][0]  = x_id
            movement[patch_id][1]  = y_id
            patch_id += 1

    return patchs, movement

if __name__ == "__main__":
    tobological_data, label = load_dataset('/Users/karino-t/repos/nupic_image_recognition/data/pylearn2_gcn_whitened/test.pkl')

    for i, data in enumerate(tobological_data):
        patch_data, movement = get_patch(data, height=5, width=5, step=3)
        print i, patch_data.shape, movement.shape, label[i]
