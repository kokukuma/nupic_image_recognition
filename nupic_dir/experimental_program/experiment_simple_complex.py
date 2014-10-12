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
        #"potentialRadius": 3,                 # default 16
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "potentialPct": 0.8,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.0001,   # default: 0.0001
        "synPermInactiveDec": 0.0005, # default: 0.0005
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
                    "type": "SDRCategoryEncoder",
                    "categoryList": ["label-" + str(x) for x in range(10)],
                    #"categoryList": ["white", "gray1", "gray2", "black"],
                    "fieldname": u"y",
                    "name": u"y",
                    "n": 102400,
                    "w": 121,
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

def deleteParam():
    # sp
    del PARAMS['SP']["columnCount"]
    del PARAMS['SP']["inputWidth"]
    del PARAMS['SP']["spatialImp"]

    # tp
    del PARAMS['TP']['columnCount']
    del PARAMS['TP']["inputWidth"]
    del PARAMS['TP']["temporalImp"]

    # cl
    del PARAMS['CL']["clVerbosity"]


def createMyNetworkSimple():

    network = myNetwork()

    # set SP
    PARAMS['SP']["inputDimensions"]             = (1024,)
    PARAMS['SP']["columnDimensions"]            = (64 * 64, )
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = 80

    network.addRegion("simplePoolerRegion", "SP", PARAMS['SP'])
    network.calc_sort.append("simplePoolerRegion")

    network.initialize()

    return network



def createMyNetworkComplex():

    network = myNetwork()

    # set SP
    PARAMS['SP']["inputDimensions"]             = (40960,) # 32 *32 * 100
    PARAMS['SP']["columnDimensions"]            = (64 * 64, )  # default 64 * 64
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = 80      # default: 80
    network.addRegion("spatialPoolerRegion", "SP", PARAMS['SP'])

    # set TP
    PARAMS['TP']['numberOfCols']   = 64 * 64            # default: 64 * 64
    PARAMS['TP']['cellsPerColumn'] = 6
    network.addRegion("temporalPoolerRegion", "TP", PARAMS['TP'])
    network.link("spatialPoolerRegion", "temporalPoolerRegion")

    # set Classifier
    PARAMS['CL']["steps"] = [0,1]
    network.addRegion( "classifierRegion", "CL", PARAMS['CL'])
    network.link("temporalPoolerRegion", "classifierRegion")

    network.initialize()

    return network

def initialize_myNetwork(simpleNetwork, myNetwork):
    SimpleRegion = simpleNetwork.regions["simplePoolerRegion"]
    SimpleRegion.setLearnmode(True)

    # Make sure learning is enabled
    SPRegion = myNetwork.regions["spatialPoolerRegion"]
    SPRegion.setLearnmode(True)


    TPRegion = myNetwork.regions["temporalPoolerRegion"]
    TPRegion.setLearnmode(True)

    return

def set_learn_myNetwork(simpleNetwork, myNetwork, enable):
    SimpleRegion = simpleNetwork.regions["simplePoolerRegion"]
    SimpleRegion.setLearnmode(enable)

    # Make sure learning is enabled
    SPRegion = myNetwork.regions["spatialPoolerRegion"]
    SPRegion.setLearnmode(enable)


    TPRegion = myNetwork.regions["temporalPoolerRegion"]
    TPRegion.setLearnmode(enable)
    return


#@profile
def runCifar10Network(train_data, train_label, simpleNetwork, complexNetwork, datanum=0, length=1000, learnMode=True):

    def toOneArray(Array):
        input_len = reduce(lambda x,y: x * y, Array.shape)
        return Array.reshape((input_len))

    global tnum

    set_learn_myNetwork(simpleNetwork, complexNetwork, learnMode)
    result = []

    patch_heigh = 32
    patch_width = 32
    patch_step  = 32

    simplePoolerRegion   = simpleNetwork.regions["simplePoolerRegion"]
    spatialPoolerRegion  = complexNetwork.regions["spatialPoolerRegion"]
    temporalPoolerRegion = complexNetwork.regions["temporalPoolerRegion"]
    classifierRegion     = complexNetwork.regions["classifierRegion"]

    prevPredictedColumns = []

    # encoder
    vector_encoder = createSensorEncoder()
    label_encoder  = createClassifierEncoder()

    datanum_i = datanum
    for i, data in enumerate(train_data[datanum:datanum+length]):
        #
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)
        label = train_label[datanum_i][0]
        label_name = "label-" + str(label)

        dataset  = []

        ## scalar_encoder
        # patch_data = [train_data[label]]
        patch_data = numpy.transpose(patch_data, (3, 0, 1,2))[0]
        image_sdr  = numpy.zeros((vector_encoder.getWidth()))
        vector_encoder.encodeIntoArray({'x': toOneArray(patch_data[0]).tolist()}, image_sdr)
        dataset.append(image_sdr)

        # simple
        simple_cell_sdr  = numpy.zeros((40960))
        for i in range(100/10):
            sum_list = numpy.zeros((64*64))
            for j in range(10):
                sidx = (i*10 + j) * 1024
                eidx = (i*10 + j + 1) * 1024
                image_patch = image_sdr[sidx:eidx]
                simpleNetwork.run(toOneArray(image_patch))
                activeColumns = simplePoolerRegion.getOutput()
                activeColumns[activeColumns == 1] = 0
                sum_list += activeColumns
            sum_list[sum_list>1] = 0            # XOR
            simple_cell_sdr[i*4096:(i+1)*4096] = numpy.clip(sum_list, 0,1)

        # ### label encoder
        # if learnMode:
        #     label_sdr = numpy.zeros((label_encoder.getWidth()))
        #     #label_sdr = numpy.zeros((vector_encoder.getWidth()))
        #     #label_sdr = numpy.zeros((32,32))
        #     label_encoder.encodeIntoArray({'y': label_name}, label_sdr)
        #     dataset.append(label_sdr)


        # complex
        for input_type, patch in enumerate([simple_cell_sdr]):
            """
            input_type == 0 : image_sdr
            input_type == 1 : label_sdr
            """
            #print len(patch.nonzero()[0] ), patch.nonzero()[0]

            # run network input
            complexNetwork.run(toOneArray(patch))

            # anomaly
            activeColumns = spatialPoolerRegion.getOutput().nonzero()[0]
            anomalyScore = computeAnomalyScore(activeColumns, prevPredictedColumns)
            prevPredictedColumns = temporalPoolerRegion.getPredictColumn().nonzero()[0]

            # Classifier
            activeCells = temporalPoolerRegion.getOutput().nonzero()[0]
            res = classifierRegion.getObj().compute(
                            recordNum=tnum,
                            patternNZ=activeCells,
                            #patternNZ=activeColumns,
                            classification={
                                'bucketIdx': createClassifierEncoder().getBucketIndices({'y': label_name})[0] if learnMode else 0,
                                'actValue': label_name if learnMode else 'no'},
                            learn=learnMode,
                            infer=True
                            )
            predict_0 = res['actualValues'][res[0].tolist().index(max(res[0]))]
            pri = "\033[32mOK\033[0m" if label_name ==  predict_0 else "\033[31mNG\033[0m"
            if learnMode:
                print '%s  y:%s  p0:%s rate:%5.2f  %s' % (datanum_i, label_name, predict_0, max(res[0]), pri)
            else:
                result.append(label_name ==  predict_0)

            tnum += 1

        datanum_i += 1

        simpleNetwork.reset()
        complexNetwork.reset()

    if not learnMode:
        print "collect count : ", result.count(True) , "/", len(result)


    return simpleNetwork, complexNetwork


if __name__ == "__main__":
    train_data, train_label = load_dataset('./data/pylearn2_test/train.pkl')
    #test_data, test_label = load_dataset('./data/pylearn2_test/test.pkl')

    deleteParam()

    simpleNetwork  = createMyNetworkSimple()
    complexNetwork = createMyNetworkComplex()
    initialize_myNetwork(simpleNetwork, complexNetwork)

    tnum    = 0
    datanum = 0
    for i in range(40):
        print
        print 'train: ' +  str(datanum)
        simpleNetwork, complexNetwork = runCifar10Network(train_data, train_label, simpleNetwork, complexNetwork, datanum, length=1000, learnMode=True)

        datanum += 5000

        print
        print 'valid: ' +  str(datanum)
        simpleNetwork, complexNetwork = runCifar10Network(train_data, train_label, simpleNetwork, complexNetwork, 49000, length=100, learnMode=False)


    # datanum = 0
    # for i in range(1):
    #     print
    #     print 'train: ' +  str(datanum)
    #     network = runCifar10Network(train_data, train_label, network, datanum, length=1000,learnMode=True)

    # datanum = 0
    # for i in range(1):
    #     print
    #     print 'train: ' +  str(datanum)
    #     network = runCifar10Network(train_data, train_label, network, datanum, length=1000,learnMode=True)
