#/usr/bin/python
# coding: utf-8

from pprint import pprint
from pylab import *
from collections import defaultdict

from nupic_dir.lib.cla_classifier import ClaClassifier
from nupic_dir.lib.function_data import function_data
from nupic_dir.lib.plotter import Plotter
from nupic_dir.lib.create_network import net_structure, sensor_params, dest_resgion_data, class_encoder_params
from nupic_dir.lib.load_data import load_dataset, get_patch


#def train(recogniter, ,label):


#@profile
def main():
    """
    """
    import random
    import numpy

    recogniter = ClaClassifier(net_structure, sensor_params, dest_resgion_data, class_encoder_params)


    tobological_data, label = load_dataset('./data/pylearn2_gcn_whitened/train.pkl')
    for i, data in enumerate(tobological_data[:1000]):
        patch_data, movement = get_patch(data, height=8, width=8, step=3)

        print '%d, label:%s, ' % (i, label[i][0]),
        for data in patch_data:
            input_len = reduce(lambda x,y: x * y, data.shape)
            input_data = {
                    'pixel': data.reshape((input_len)).tolist() ,
                    'label': label[i][0]
                    }
            if i > 900:
                inferences = recogniter.run(input_data, learn=True, class_learn=True, learn_layer=None)
            else:
                inferences = recogniter.run(input_data, learn=True, class_learn=False, learn_layer=None)

            recogniter.print_inferences(input_data, inferences)
        recogniter.reset()

    tobological_data, label = load_dataset('./data/pylearn2_gcn_whitened/test.pkl')
    for i, data in enumerate(tobological_data):
        patch_data, movement = get_patch(data, height=8, width=8, step=3)

        print '%d, label:%s, ' % (i, label[i][0]),
        for data in patch_data:
            input_len = reduce(lambda x,y: x * y, data.shape)
            input_data = {
                    'pixel': data.reshape((input_len)).tolist() ,
                    'label': 'no'
                    }
            inferences = recogniter.run(input_data, learn=False, class_learn=False,learn_layer=None)

            recogniter.print_inferences(input_data, inferences)
        recogniter.reset()


    # # トレーニング
    # #for learn_layer in [['region1'], ]:
    # for num, ftype in enumerate(fd.function_list.keys()):
    #     data  = fd.get_data(ftype)
    #     label = fd.get_label(ftype)
    #     for x, y in data:
    #         input_data = {
    #                 'xy_value': [x, y],
    #                 'x_value': x,
    #                 'y_value': y,
    #                 'ftype': label
    #                 }
    #
    #         inferences = recogniter.run(input_data, learn=True, learn_layer=learn_layer)
    #
    #         # print
    #         recogniter.print_inferences(input_data, inferences)
    #     recogniter.reset()

if __name__ == "__main__":
    main()
