#/usr/bin/python
# coding: utf-8

"""
cifar training by opf
"""

from collections import defaultdict, Counter

from nupic_dir.lib.load_data import load_dataset, get_patch
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params

test_data, test_label   = load_dataset('./data/pylearn2_gcn_whitened/test.pkl')
train_data, train_label = load_dataset('./data/pylearn2_gcn_whitened/train.pkl')
patch_heigh = 3
patch_width = 3
patch_step  = 3

def validate(model, test_data, test_label, limit=100):
    result = []
    tdata = test_data[:limit]
    for i, data in enumerate(tdata):
        if test_label[i][0] not in (0, 1):
            continue
        patch_result = Counter()
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)

        for patch in patch_data:
            input_len = reduce(lambda x,y: x * y, patch.shape)
            input_data = {
                    'pixel': patch.reshape((input_len)).tolist() ,
                    'label': 'no'
                    }
            model.disableLearning()
            modelresult = model.run(input_data)
            #print label[i][0], result.inferences['multiStepBestPredictions']


            best_result = modelresult.inferences['multiStepBestPredictions']
            #patch_result[best_result['value']] += best_result['prob']
            patch_result[best_result[0]] += 1

        # print test_label[i][0]
        # print patch_result

        if test_label[i][0] == max(patch_result.items(), key=lambda x:x[1])[0]:
            result.append(1)

        #model.resetSequenceStates()
        model._getTPRegion().getSelf().resetSequenceStates()

    return len(result)/len(tdata)


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
    for i, data in enumerate(tobological_data[:1000]):
        if label[i][0] not in (0, 1):
            continue

        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)

        print '%d, label:%s, ' % (i, label[i][0]),
        for data in patch_data:
            input_len = reduce(lambda x,y: x * y, data.shape)
            input_data = {
                    'pixel': data.reshape((input_len)).tolist() ,
                    'label': label[i][0]
                    }

            model.enableLearning()
            result = model.run(input_data)

            #result = shifter.shift(result)
            print label[i][0], result.inferences['multiStepBestPredictions']

        #model.resetSequenceStates()
        #model._getTPRegion().executeCommand(['resetSequenceStates'])
        #model._getTPRegion().resetSequenceStates()
        model._getTPRegion().getSelf().resetSequenceStates()


        # validate
        if i % 3 == 0:
            valid = validate(model, test_data, test_label, limit=30)
            print '%d : valid: %8.5f' % (i, valid)


if __name__ == "__main__":
    main()
