#/usr/bin/python
# coding: utf-8

from pprint import pprint
from pylab import *
from collections import defaultdict, Counter

from nupic_dir.lib.cla_classifier import ClaClassifier
from nupic_dir.lib.function_data import function_data
from nupic_dir.lib.plotter import Plotter
from nupic_dir.lib.model import net_structure, sensor_params, dest_resgion_data, class_encoder_params
from nupic_dir.lib.load_data import load_dataset, get_patch


test_data, test_label   = load_dataset('./data/pylearn2_gcn_whitened/test.pkl')
train_data, train_label = load_dataset('./data/pylearn2_gcn_whitened/train.pkl')
patch_heigh = 32
patch_width = 32
patch_step  = 32


def validate(recogniter, test_data, test_label, limit=100):
    result = []
    tdata = test_data[:limit]
    for i, data in enumerate(tdata):
        patch_result = Counter()
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)

        for patch in patch_data:
            input_len = reduce(lambda x,y: x * y, patch.shape)
            input_data = {
                    'pixel': patch.reshape((input_len)).tolist() ,
                    'label': 'no'
                    }
            inferences = recogniter.run(input_data, learn=True, class_learn=False,learn_layer=None)


            best_result = inferences['classifier_region1']['best']
            #patch_result[best_result['value']] += best_result['prob']
            patch_result[best_result['value']] += 1

        # print test_label[i][0]
        # print patch_result

        if test_label[i][0] == max(patch_result.items(), key=lambda x:x[1])[0]:
            result.append(1)

        recogniter.reset()

    return float(len(result))/len(tdata)


#@profile
def main():
    """
    """
    import random
    import numpy

    recogniter = ClaClassifier(net_structure, sensor_params, dest_resgion_data, class_encoder_params)

    print 'training ...'
    for i, data in enumerate(train_data[:1000]):

        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)

        for patch in patch_data:
            input_len = reduce(lambda x,y: x * y, patch.shape)
            input_data = {
                    'pixel': patch.reshape((input_len)).tolist() ,
                    'label': train_label[i][0]
                    }
            inferences = recogniter.run(input_data, learn=True, class_learn=True, learn_layer=None)
            #recogniter.layer_output(input_data)
            #recogniter.print_inferences(input_data, inferences)

        #print train_label[i][0] , inferences['classifier_region1']['best']

        recogniter.reset()

        # validate
        if i % 50 == 0 and not i == 0:
            valid = validate(recogniter, test_data, test_label, limit=30)
            print '%d : valid: %8.5f' % (i, valid)

if __name__ == "__main__":
    main()
