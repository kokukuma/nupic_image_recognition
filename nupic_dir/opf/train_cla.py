#/usr/bin/python
# coding: utf-8

"""
cifar training by opf
"""

from nupic_dir.lib.load_data import load_dataset, get_patch
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params

def createModel():
    model = ModelFactory.create(model_params.config)
    model.enableInference({
        "predictedField": "label"
        })
    return model

#@profile
def main():
    """
    """
    import random
    import numpy

    model = createModel()
    shifter = InferenceShifter()


    tobological_data, label = load_dataset('./data/pylearn2_gcn_whitened/train.pkl')
    for i, data in enumerate(tobological_data[:200]):
        patch_data, movement = get_patch(data, height=3, width=3, step=3)

        print '%d, label:%s, ' % (i, label[i][0]),
        for data in patch_data:
            input_len = reduce(lambda x,y: x * y, data.shape)
            input_data = {
                    'pixel': data.reshape((input_len)).tolist() ,
                    'label': label[i][0]
                    }
            result = model.run(input_data)

            result = shifter.shift(result)
            print label[i][0], result.inferences['multiStepBestPredictions']


    tobological_data, label = load_dataset('./data/pylearn2_gcn_whitened/test.pkl')
    for i, data in enumerate(tobological_data):
        patch_data, movement = get_patch(data, height=3, width=3, step=3)

        print '%d, label:%s, ' % (i, label[i][0]),
        for data in patch_data:
            input_len = reduce(lambda x,y: x * y, data.shape)
            input_data = {
                    'pixel': data.reshape((input_len)).tolist() ,
                    'label': 'no'
                    }
            result = model.run(input_data)

            result = shifter.shift(result)
            print label[i][0], result.inferences['multiStepBestPredictions']


if __name__ == "__main__":
    main()
