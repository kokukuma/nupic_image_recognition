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


def stochastic_encoder(inputArray, minVal=0, maxVal=255, sample_number=20):
    """
    inputArray : (32,32)
    return : (32,32)
    """
    result = []
    numInput = reduce(lambda x,y: x*y, inputArray.shape)
    prob   = inputArray.reshape(numInput).astype("float32") / maxVal

    sample = numpy.ones(numInput)
    for i in range(sample_number):
        for j, p in enumerate(prob):
            sample[j] = numpy.random.binomial(n=1, p=p)
        result.append(sample.reshape(inputArray.shape) )
    return result

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
                    "n": 50,
                    "w": 13,
                    "clipInput": True,
                    "forced": True
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
        "synPermActiveInc": 0.01,   # default: 0.0001
        "synPermInactiveDec": 0.05, # default: 0.0005
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
                    "length": 64,
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
    PARAMS['SP']["inputDimensions"]             = (50, 32, 32)
    PARAMS['SP']["columnDimensions"]            = (10, 8, 8)
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = int(10. * 8. * 8. * 0.02)
    network.addRegion("simplePoolerRegion", "SP", PARAMS['SP'])
    network.calc_sort.append("simplePoolerRegion")

    network.initialize()

    return network

def createMyNetworkSimple2():

    network = myNetwork()

    # set SP
    PARAMS['SP']["inputDimensions"]             = (25, 16, 16)
    PARAMS['SP']["columnDimensions"]            = (10, 8, 8)
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = int(5. * 8. * 8. * 0.02)
    network.addRegion("simplePoolerRegion", "SP", PARAMS['SP'])
    network.calc_sort.append("simplePoolerRegion")

    network.initialize()

    return network


def createMyNetworkSimple3():

    network = myNetwork()

    # set SP
    PARAMS['SP']["inputDimensions"]             = (10, 24, 24)
    PARAMS['SP']["columnDimensions"]            = (5, 8, 8)
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = int(15. * 8. * 8. * 0.02)
    network.addRegion("simplePoolerRegion", "SP", PARAMS['SP'])
    network.calc_sort.append("simplePoolerRegion")

    network.initialize()

    return network


def createMyNetworkComplex():

    network = myNetwork()

    # set SP
    PARAMS['SP']["inputDimensions"]             = (27034,)
    PARAMS['SP']["columnDimensions"]            = (2048,)
    PARAMS['SP']["numActiveColumnsPerInhArea"]  = int(2048. * 0.02)
    network.addRegion("simplePoolerRegion", "SP", PARAMS['SP'])

    # set TP
    PARAMS['TP']['numberOfCols']   = 2048            # default: 64 * 64
    PARAMS['TP']['cellsPerColumn'] = 32
    network.addRegion("temporalPoolerRegion", "TP", PARAMS['TP'])
    network.link("simplePoolerRegion", "temporalPoolerRegion")

    # set Classifier
    PARAMS['CL']["steps"] = [0,1]
    network.addRegion( "classifierRegion", "CL", PARAMS['CL'])
    #network.link("temporalPoolerRegion", "classifierRegion")

    network.initialize()

    return network

def initialize_myNetwork(simpleNetwork):
    SimpleRegion = simpleNetwork.regions["simplePoolerRegion"]
    SimpleRegion.setLearnmode(True)
    return

def set_learn_myNetwork(simpleNetwork, enable):
    SimpleRegion = simpleNetwork.regions["simplePoolerRegion"]
    SimpleRegion.setLearnmode(enable)
    return


#@profile
def runCifar10Network(train_data, train_label, simpleNetwork, simple2Network, simple3Network, complexNetwork, datanum=0, length=1000, learnMode=[True]):

    def toOneArray(Array):
        input_len = reduce(lambda x,y: x * y, Array.shape)
        return Array.reshape((input_len))

    global tnum

    classifierRegion     = complexNetwork.regions["classifierRegion"]

    set_learn_myNetwork(simpleNetwork, learnMode[0])
    set_learn_myNetwork(simple2Network, learnMode[1])
    set_learn_myNetwork(simple3Network, learnMode[2])
    classifier_learnMode = learnMode[3]

    result = []

    patch_heigh = 32
    patch_width = 32
    patch_step  = 32

    #
    label_patch_data = []
    base_sdr = None

    def calc_network(network, patch):
        network.run(toOneArray(patch))
        simplePoolerRegion = network.regions["simplePoolerRegion"]
        return simplePoolerRegion.getOutput()

    def fusion(activeset, odim):
        patch_list = numpy.zeros(odim)
        for num, res in enumerate(list(numpy.transpose(activeset, (1, 0, 2, 3)))):
            patch_list[num] = numpy.r_[ numpy.c_[res[0], res[1]],
                                        numpy.c_[res[2], res[3]]]
        return patch_list

    def fusion2(activeset, odim):
        patch_list = numpy.zeros(odim)
        for num, res in enumerate(list(numpy.transpose(activeset, (1, 0, 2, 3)))):
            patch_list[num] = numpy.r_[ numpy.c_[res[0], res[1] ,res[2]],
                                        numpy.c_[res[3], res[4], res[5]],
                                        numpy.c_[res[6], res[7], res[8]]]
        return patch_list


    datanum_i = datanum
    for i, data in enumerate(train_data[datanum:datanum+length]):
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)
        label_name = "label-" + str(train_label[datanum_i][0] )

        dataset = temporal_scalor_encoder(patch_data[0])
        activeColumns = calc_network(simpleNetwork, dataset)

        # print activeColumns.shape
        # print activeColumns.reshape((10, 8, 8))

        for idx in activeColumns.nonzero()[0]:

            # simpleNetwork
            simple_sp = simpleNetwork.regions["simplePoolerRegion"].getObj()
            mask = simple_sp.mapPotential_(idx, True)
            mask = numpy.asarray(mask).nonzero()[0]
            patch = toOneArray(dataset)[mask]
            #print patch.shape, patch,

            # complexNetwork
            complexNetwork.run(toOneArray(patch))
            temporalPoolerRegion = complexNetwork.regions["temporalPoolerRegion"]
            activeCells = temporalPoolerRegion.getOutput()

            print idx, activeCells.nonzero()[0]


        complexNetwork.reset()
        #print activeCells.nonzero()[0]



        # activeset_16 = numpy.zeros((9, 5, 8, 8))
        # for num_16, patch_16 in enumerate(patch_data):
        #
        #     activeset_8 = numpy.zeros((9, 10, 8, 8))
        #     patch_data_8, movement = get_patch(patch_16, height=8, width=8, step=4)
        #     for num_8, patch_8 in enumerate(patch_data_8):
        #
        #         activeset_4 = numpy.zeros((4, 25, 8, 8))  #activeset = numpy.zeros((4, 5, 8, 8))
        #         patch_data_4, movement = get_patch(patch_8, height=4, width=4, step=4)
        #         for num_4, patch in enumerate(patch_data_4):
        #             """
        #             第１ネットワーク (20, 4, 4) -> (5, 8, 8)
        #             第１ネットワーク (50, 4, 4) -> (25, 8, 8)
        #             """
        #             dataset = temporal_scalor_encoder(patch)
        #
        #             activeColumns = calc_network(simpleNetwork, dataset)
        #             activeset_4[num_4] = activeColumns.reshape((25, 8, 8))
        #
        #         patch_list_4 = fusion(activeset_4, odim=(25, 16, 16))
        #
        #         """
        #         第２ネットワーク (5, 8, 8) -> (5, 8, 8)
        #         第２ネットワーク (25, 8, 8) -> (10, 8, 8)
        #         """
        #         activeColumns = calc_network(simple2Network, patch_list_4)
        #         activeset_8[num_8] = activeColumns.reshape((10, 8, 8))
        #
        #     patch_list_8 = fusion2(activeset_8, odim=(10, 24, 24))
        #
        #     """
        #     第３ネットワーク(5, 24, 24) -> (5, 8, 8)
        #     """
        #     # dataset = temporal_scalor_encoder(patch_16)
        #     # activeColumns = calc_network(simple3Network, dataset)
        #     activeColumns = calc_network(simple3Network, patch_list_8)
        #     activeset_16[num_16] = activeColumns.reshape((5, 8, 8))
        #
        #
        # """
        # 第４ネットワーク(5, 24, 24) -> (5, 8, 8)
        # """
        # patch_list_16 = fusion2(activeset_16, odim=(5, 24, 24))
        # activeColumns = toOneArray(patch_list_16)
        # #activeColumns = calc_network(complexNetwork, patch_list_16)


            # """
            # 結果表示
            # """
            # # if 4 in activeColumns.nonzero()[0]:
            # #     label_patch_data.append(patch2)
            # #     print datanum_i, pnum, activeColumns.nonzero()[0]
            # if num_16>= 0:
            #     if base_sdr == None:
            #         base_patch = patch_16
            #         base_sdr = activeColumns.nonzero()[0]
            #         print 'base : ', label_name, base_sdr
            #     else:
            #         if len(set(base_sdr) & set(activeColumns.nonzero()[0]))  >= len(base_sdr)/1.5:
            #             print datanum_i, label_name, num_16, activeColumns.nonzero()[0]
            #             label_patch_data.append(patch_16)
        """
        結果取得
        """
        # if 4 in activeColumns.nonzero()[0]:
        #     label_patch_data.append(patch2)
        #     print datanum_i, pnum, activeColumns.nonzero()[0]
        if i>= 0:
            if base_sdr == None:
                base_patch = data
                base_sdr = activeCells.nonzero()[0]
                #print 'base : ', label_name, base_sdr
            else:
                if len(set(base_sdr) & set(activeCells.nonzero()[0]))  >= len(base_sdr)/2:
                    #print datanum_i, label_name, i, activeCells.nonzero()[0]
                    label_patch_data.append(data)

        """
        classifier
        """
        #activeCells = activeColumns.nonzero()[0]
        res = classifierRegion.getObj().compute(
                        recordNum=tnum,
                        patternNZ=activeCells.nonzero()[0] ,
                        classification={
                            'bucketIdx': createClassifierEncoder().getBucketIndices({'y': label_name})[0] if classifier_learnMode  else 0,
                            'actValue': label_name if classifier_learnMode else 'no'},
                        learn=classifier_learnMode,
                        infer=True
                        )
        predict_0 = res['actualValues'][res[0].tolist().index(max(res[0]))]
        pri = "\033[32mOK\033[0m" if label_name ==  predict_0 else "\033[31mNG\033[0m"
        if classifier_learnMode:
            print '%s  y:%s  p0:%s rate:%5.2f  %s' % (datanum_i, label_name, predict_0, max(res[0]), pri)
        else:
            result.append(label_name ==  predict_0)

        tnum += 1

        datanum_i += 1
        continue

    """
    画像表示
    """
    from PIL import Image
    img = Image.new('L', (160, 160))
    numpy.random.shuffle(label_patch_data)   # 同じ画像内ばかりでないようにshuffle.
    label_patch_data.insert(0, base_patch)   # 左上がbase_patchになるように.
    for i, patch in enumerate(label_patch_data[:25]):
        image_patch = patch.reshape((32,32))
        for a in range(32):
            for b in range(32):
                aidx = (i * 32 + a)  % 160
                bidx = (i / 5) * 32 + b
                img.putpixel((bidx,aidx), image_patch[a][b])
    img.show()


    if not classifier_learnMode:
        print "collect count : ", result.count(True) , "/", len(result)


    return simpleNetwork, simple2Network, simple3Network


if __name__ == "__main__":
    train_data, train_label = load_dataset('./data/pylearn2_test/train.pkl')
    #test_data, test_label = load_dataset('./data/pylearn2_test/test.pkl')

    deleteParam()

    print 'make network'
    simpleNetwork   = createMyNetworkSimple()
    simple2Network  = createMyNetworkSimple2()
    simple3Network  = createMyNetworkSimple3()

    complexNetwork = createMyNetworkComplex()

    print 'init network'
    initialize_myNetwork(simpleNetwork)
    initialize_myNetwork(simple2Network)
    initialize_myNetwork(simple3Network)

    tnum    = 0
    datanum = 0
    for i in range(40):
        # print
        # print 'train1: ' +  str(datanum)
        # tmp_datanum = datanum
        # simpleNetwork, simple2Network, simple3Network = \
        #     runCifar10Network(train_data, train_label, simpleNetwork, simple2Network, simple3Network, complexNetwork, datanum, length=1000, learnMode=[True, False, False, False])
        # datanum = tmp_datanum
        #
        # print 'train2: ' +  str(datanum)
        # simpleNetwork, simple2Network, simple3Network = \
        #     runCifar10Network(train_data, train_label, simpleNetwork, simple2Network, simple3Network, complexNetwork, datanum, length=1000, learnMode=[False, True, False, False])
        # datanum = tmp_datanum

        print 'train3: ' +  str(datanum)
        simpleNetwork, simple2Network, simple3Network = \
            runCifar10Network(train_data, train_label, simpleNetwork, simple2Network, simple3Network, complexNetwork,datanum, length=100, learnMode=[True, True, True, True])

        datanum += 100

        print
        print 'valid: ' +  str(datanum)
        simpleNetwork, simple2Network, simple3Network = \
            runCifar10Network(train_data, train_label, simpleNetwork, simple2Network, simple3Network, complexNetwork, 49000, length=100, learnMode=[False, False, False, False])

