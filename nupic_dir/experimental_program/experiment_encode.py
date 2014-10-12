#!/usr/bin/env python
# encoding: utf-8
import copy
import json
import numpy
from collections import defaultdict, Counter

from nupic.algorithms.anomaly import computeAnomalyScore
from nupic.engine import Network
from nupic.encoders import MultiEncoder

from netowrk import Network as myNetwork

from nupic_dir.lib.load_data import load_dataset, get_patch


def get_color_data():
    sample_number = 50

    dataset = {}
    # dataset['white'] =  numpy.asarray([[  0,  0,  0,  0,  0,  0],
    #                                    [  0,  0,  0,  0,  0,  0],
    #                                    [  0,  0,  0,  0,  0,  0],
    #                                    [  0,  0,  0,  0,  0,  0],
    #                                    [  0,  0,  0,  0,  0,  0],
    #                                    [  0,  0,  0,  0,  0,  0]])
    # dataset['gray1'] =  numpy.asarray([[ 80, 80, 80, 80, 80, 80],
    #                                    [ 80, 80, 80, 80, 80, 80],
    #                                    [ 80, 80, 80, 80, 80, 80],
    #                                    [ 80, 80, 80, 80, 80, 80],
    #                                    [ 80, 80, 80, 80, 80, 80],
    #                                    [ 80, 80, 80, 80, 80, 80]])
    # dataset['gray2'] =  numpy.asarray([[160,160,160,160,160,160],
    #                                    [160,160,160,160,160,160],
    #                                    [160,160,160,160,160,160],
    #                                    [160,160,160,160,160,160],
    #                                    [160,160,160,160,160,160],
    #                                    [160,160,160,160,160,160]])
    # dataset['black'] =  numpy.asarray([[255,255,255,255,255,255],
    #                                    [255,255,255,255,255,255],
    #                                    [255,255,255,255,255,255],
    #                                    [255,255,255,255,255,255],
    #                                    [255,255,255,255,255,255],
    #                                    [255,255,255,255,255,255]])
    dataset['white'] =  numpy.asarray([[  0,  0,  0,  0],
                                       [  0,  0,  0,  0],
                                       [  0,  0,  0,  0],
                                       [  0,  0,  0,  0]])
    dataset['gray1'] =  numpy.asarray([[ 80, 80, 80, 80],
                                       [ 80, 80, 80, 80],
                                       [ 80, 80, 80, 80],
                                       [ 80, 80, 80, 80]])
    dataset['gray2'] =  numpy.asarray([[160,160,160,160],
                                       [160,160,160,160],
                                       [160,160,160,160],
                                       [160,160,160,160]])
    dataset['black'] =  numpy.asarray([[255,255,255,255],
                                       [255,255,255,255],
                                       [255,255,255,255],
                                       [255,255,255,255]])
    # dataset['white'] =  numpy.asarray([[  0,  0],
    #                                    [  0,  0]])
    # dataset['gray1'] =  numpy.asarray([[ 80, 80],
    #                                    [ 80, 80]])
    # dataset['gray2'] =  numpy.asarray([[160,160],
    #                                    [160,160]])
    # dataset['black'] =  numpy.asarray([[255,255],
    #                                    [255,255]])
    # dataset['white'] =  numpy.asarray([[  0,  0],
    #                                    [  0,  0]])
    # dataset['gray1'] =  numpy.asarray([[ 80, 80],
    #                                    [ 80, 80]])
    # dataset['gray2'] =  numpy.asarray([[160,160],
    #                                    [160,160]])
    # dataset['black'] =  numpy.asarray([[255,255],
    #                                    [255,255]])
    result = defaultdict(list)
    for label, data in dataset.items():
        print '--- ' + label
        for i in range(sample_number):
            print stochastic_encoder(data)
            result[label].append(stochastic_encoder(data))
    return result


def stochastic_encoder(inputArray, minVal=0, maxVal=255):
    """
    inputArray : (32,32)
    return : (32,32)
    """
    numInput = reduce(lambda x,y: x*y, inputArray.shape)
    prob   = inputArray.reshape(numInput).astype("float32") / maxVal

    sample = numpy.ones(numInput)
    for i, p in enumerate(prob):
        sample[i] = numpy.random.binomial(n=1, p=p)
    return sample.reshape(inputArray.shape)

def temporal_scalor_encoder(inputArray, minVal=0, maxVal=255):
    """
    inputArray : (32,32)
    return : (n, 32,32)
    """

    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
            "x": {
                    "type": "ScalarEncoder",
                    "fieldname": u"x",
                    "name": u"x",
                    "maxval": 255.0,
                    "minval": 0.0,
                    "n": 36,
                    "w": 21,
                    "clipInput": True,
            },
    })
    numInput   = reduce(lambda x,y: x*y, inputArray.shape)

    result = numpy.ones([numInput, encoder.getWidth()])

    for i, value in enumerate(inputArray.reshape(numInput)):
        encoder.encodeIntoArray({'x': value}, result[i])

    result = numpy.transpose(result, (1,0))  # [numInput, encoder.getWidth()] -> [encoder.getWidth(), numInput]
    result = result.reshape( (encoder.getWidth(), inputArray.shape[0], inputArray.shape[1]))
    return result


PARAMS = {
    'SP': {
        "spVerbosity": 0,
        "spatialImp": "cpp",
        "globalInhibition": 1,
        "columnCount": 2048,
        #"columnCount": 2048, SP(32, 32) -> TP(1024)
        # This must be set before creating the SPRegion
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "potentialPct": 0.8,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.0001,
        "synPermInactiveDec": 0.0005,
        "maxBoost": 1.0,
    },
    'TP':{
        "verbosity": 0,
        "columnCount": 2048,
        "cellsPerColumn": 32,
        "inputWidth": 2048,
        "seed": 1960,
        "collectStats": True,
        "temporalImp": "cpp",
        "newSynapseCount": 20,
        "maxSynapsesPerSegment": 32,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 9,
        "activationThreshold": 12,
        "outputType": "normal",
        "pamLength": 3,
    },
    'CL':{
        "clVerbosity": 0,
        "alpha": 0.005,
        "steps": "0"
    },
}

def createSensorEncoder():
    """Create the encoder instance for our test and return it."""
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
            "x": {
                    #"type": "ScalarEncoder",
                    "type": "VectorEncoderOPF",
                    "length": 1024,
                    "fieldname": u"x",
                    "name": u"x",
                    "maxval": 255.0,
                    "minval": 0.0,
                    "n": 100,
                    "w": 21,
                    "clipInput": True,
            },
    })

    return encoder

def createClassifierEncoder():
    """Create the encoder instance for our test and return it."""
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
            "y": {
                    "type": "CategoryEncoder",
                    "categoryList": range(10),
                    #"categoryList": ["white", "gray1", "gray2", "black"],
                    "fieldname": u"y",
                    "name": u"y",
                    "w": 21,
            },
    })

    return encoder

"""
--------------------------------------------
Nupic Network
--------------------------------------------
"""

"""
実験内容
1. 2次元SPをした場合としない場合で結果に違いが出るか?
   2次元SPの法が良い結果になることはあるか?

  + 縦線と横線の判別.
  + 縦線と横線の動きの判別.

2. 確率的エンコーダーを利用して, 色の違いを判別できるか?
  + 9x9, １色の判別.
"""

def changeArg():
    # sp
    del PARAMS['SP']["columnCount"]
    del PARAMS['SP']["inputWidth"]
    del PARAMS['SP']["spatialImp"]
    #PARAMS['SP']["inputDimensions"]             = (123904, )  # 32 *32 * 121
    # PARAMS['SP']["inputDimensions"]             = (36864, )  # 32 *32 * 36

    # PARAMS['SP']["inputDimensions"]             = (102400,) # 32 *32 * 100
    # PARAMS['SP']["columnDimensions"]            = (4096, )

    PARAMS['SP']["inputDimensions"]           = (32, 32)
    PARAMS['SP']["columnDimensions"]          = (64, 64)

    PARAMS['SP']["numActiveColumnsPerInhArea"]  = 80

    # tp
    PARAMS['TP']['numberOfCols']   = 64 * 64
    PARAMS['TP']['cellsPerColumn'] = 2

    del PARAMS['TP']['columnCount']
    del PARAMS['TP']["inputWidth"]
    del PARAMS['TP']["temporalImp"]

    # cl
    del PARAMS['CL']["clVerbosity"]
    PARAMS['CL']["steps"] = [0]

def createMyNetwork():
    changeArg()

    network = myNetwork()

    # set SP
    network.addRegion("spatialPoolerRegion", "SP", PARAMS['SP'])

    # set TP
    network.addRegion("temporalPoolerRegion", "TP", PARAMS['TP'])
    network.link("spatialPoolerRegion", "temporalPoolerRegion")

    # set Classifier
    network.addRegion( "classifierRegion", "CL", PARAMS['CL'])
    network.link("temporalPoolerRegion", "classifierRegion")

    network.initialize()

    return network

def initialize_myNetwork(myNetwork):
    # Make sure learning is enabled
    SPRegion = myNetwork.regions["spatialPoolerRegion"]
    SPRegion.setLearnmode(True)

    TPRegion = myNetwork.regions["temporalPoolerRegion"]
    TPRegion.setLearnmode(True)
    return network


#@profile
def runCifar10Network(train_data, train_label, network, datanum=0, length=1000, learnMode=True, printDebug=True):
    global tnum

    patch_heigh = 32
    patch_width = 32
    patch_step  = 32

    spatialPoolerRegion  = network.regions["spatialPoolerRegion"]
    temporalPoolerRegion = network.regions["temporalPoolerRegion"]
    classifierRegion     = network.regions["classifierRegion"]

    prevPredictedColumns = []

    # count
    other_active_count = Counter()
    cell_active_count = {}
    for i in range(10):
        cell_active_count[i] = Counter()

    # encoder
    vector_encoder = createSensorEncoder()

    for i, data in enumerate(train_data[datanum:datanum+length]):
        #print '================= ' + str(i)
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)
        label = train_label[i][0]


        ## patch分割
        # patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0] / 255
        # for patch in patch_data:

        # # 繰り返し
        # patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0] / 255
        # #for patch in patch_data:
        # for i in range(1):
        #     patch = patch_data[0]

        # ## stochastic_encoder + patch分割
        # for patch_4x4 in patch_data:
        #     patch_4x4 = numpy.transpose(patch_4x4, (2, 0, 1))[0]
        #     for i in range(10):
        #         # 0,255 -> 0,1
        #         patch = stochastic_encoder(patch_4x4)

        ## stochastic_encoder
        # patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0]
        # for j in range(10):
        #     # 0,255 -> 0,1
        #     patch = stochastic_encoder(patch_data[0])

        # ## scalar_encoder
        # patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0]
        # for j in range(1):
        #     # 0,255 -> 0,1
        #     patch_data =  patch_data[0].reshape(reduce(lambda x,y: x*y, patch_data[0].shape))
        #     patch = numpy.zeros((vector_encoder.getWidth()))
        #     vector_encoder.encodeIntoArray({'x': patch_data.tolist()}, patch)

        # temporal_scalor_encoder
        patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0]
        for patch in temporal_scalor_encoder(patch_data[0]):

            input_len = reduce(lambda x,y: x * y, patch.shape)

            network.run(patch.reshape((input_len)))

            # anomaly
            activeColumns = spatialPoolerRegion.getOutput().nonzero()[0]
            anomalyScore = computeAnomalyScore(activeColumns, prevPredictedColumns)
            prevPredictedColumns = temporalPoolerRegion.getPredictColumn().nonzero()[0]

            # Classifier
            activeCells = temporalPoolerRegion.getOutput().nonzero()[0]


        for cell in activeCells:
            cell_active_count[label][cell] += 1

        if label == 0:
            print i, anomalyScore, cell_active_count[label].most_common(5)

            # res = classifierRegion.getObj().compute(
            #                 recordNum=tnum,
            #                 patternNZ=activeCells,
            #                 #patternNZ=activeColumns,
            #                 classification={
            #                     'bucketIdx': createClassifierEncoder().getBucketIndices({'y': label})[0] if learnMode else 0,
            #                     'actValue': label if learnMode else 'no'},
            #                 learn=learnMode,
            #                 infer=True
            #                 )
            # predict = res['actualValues'][res[0].tolist().index(max(res[0]))]
            #
            # # print
            # if label ==  predict:
            #     pri = "\033[32mOK\033[0m"
            # else:
            #     pri = "\033[31mNG\033[0m"
            # if printDebug:
            #     print '%s  y:%s  p:%s  rate:%5.2f  anomaly:%5.2f  %s' % (tnum, label, predict, max(res[0]), anomalyScore, pri)
            #
            #     tnum += 1
            #
        network.reset()

    def print_summary(counter, label):
        other = []
        for i in range(10):
            if not i == label:
                other += counter[i].keys()

        print "    ", len(counter[label]), len(set(counter[0].keys()) - set(other))

    def top_active_print(counter, label, limit=10):
        other_counter = Counter()
        for i in range(10):
            if not i == label:
                other_counter += counter[i]

        for i, data in enumerate([x for x in counter[label].most_common(limit)]):
            print "    ",i, data[0], data[1], other_counter[data[0]]

    def print_plot(counterset):
        from plotter import Plotter
        plotter    = Plotter()
        plotter.initialize({
            'active':{
                'ylim': [0,1],
                'sub_title': counterset.keys()},
            }, movable=False)

        x_values = {}
        y_values = {}
        for label, counter in counterset.items():
            data = defaultdict(int)
            x_value = []
            y_value = []
            for i in counter.most_common(1000000):
                data[i[1]] += 1
            for x in range(30):
                x_value.append(x)
                y_value.append(data[x])
            x_values[label] = x_value
            y_values[label] = [float(y) / sum(y_value)for y in y_value]
            print label, x_value, y_value
        plotter.add(title="active", x_values=x_values, y_values=y_values)
        plotter.show()


    print "=========== OR summary ==========="
    for i in range(10):
        print "label : ", i,
        print_summary(cell_active_count, i)

    print "=========== top 10 ==========="
    for i in range(10):
        print "label : ", i
        top_active_print(cell_active_count, i, limit=10)

    print "=========== plot ==========="
    for i in range(10000):
        print i,
        for j in range(10):
            if cell_active_count[j][i] == 0:
                print ' ',
            else:
                print cell_active_count[j][i],
        print

    print "=========== plot ==========="
    other_counter = Counter()
    for i in range(10):
        other_counter += cell_active_count[i]
    cell_active_count['all'] = other_counter
    print_plot(cell_active_count)


    # # MEMO: label 0 のtop 30のセルは他のラベルでも発火している...
    # print "label 0"
    # for i, data in enumerate([x for x in cell_active_count.most_common(30)]):
    #     print i, data[0], data[1], other_active_count[data[0]]
    # print
    # print "other"
    # for i, data in enumerate([x for x in other_active_count.most_common(30)]):
    #     print i, data
    # print

    return network


if __name__ == "__main__":
    train_data, train_label = load_dataset('./data/pylearn2_test/train.pkl')
    #test_data, test_label = load_dataset('./data/pylearn2_test/test.pkl')

    network = createMyNetwork()
    initialize_myNetwork(network)

    tnum    = 0
    datanum = 0
    for i in range(1):
        print
        print 'train: ' +  str(datanum)
        network = runCifar10Network(train_data, train_label, network, datanum, length=1000, learnMode=True, printDebug=True)

    # datanum = 0
    # for i in range(1):
    #     print
    #     print 'train: ' +  str(datanum)
    #     network = runCifar10Network(train_data, train_label, network, datanum, length=1000,learnMode=True, printDebug=True)
    #
    # datanum = 0
    # for i in range(1):
    #     print
    #     print 'train: ' +  str(datanum)
    #     network = runCifar10Network(train_data, train_label, network, datanum, length=1000,learnMode=True, printDebug=True)
