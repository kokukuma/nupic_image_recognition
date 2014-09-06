#/usr/bin/python
# coding: utf-8

from pprint import pprint
from pylab import *
from collections import defaultdict

from nupic_dir.lib.cla_classifier import ClaClassifier
from nupic_dir.lib.function_data import function_data
from nupic_dir.lib.plotter import Plotter
from nupic_dir.lib.create_network import net_structure, sensor_params, dest_resgion_data, class_encoder_params

def main():
    import random
    import numpy

    fd = function_data()
    recogniter = ClaClassifier(net_structure, sensor_params, dest_resgion_data, class_encoder_params)

    # トレーニング
    for learn_layer in [['region1'], ]:
        for i in range(5):
            print i,
            for num, ftype in enumerate(fd.function_list.keys()):
                data  = fd.get_data(ftype)
                label = fd.get_label(ftype)
                for x, y in data:
                    input_data = {
                            'xy_value': [x, y],
                            'x_value': x,
                            'y_value': y,
                            'ftype': label
                            }

                    inferences = recogniter.run(input_data, learn=True, learn_layer=learn_layer)

                    # print
                    recogniter.print_inferences(input_data, inferences)
                recogniter.reset()

if __name__ == "__main__":
    main()
