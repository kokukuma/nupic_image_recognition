#!/usr/bin/python
# coding: utf-8


import numpy
import json
import itertools
import copy

from collections import defaultdict
from collections import OrderedDict

from nupic.algorithms.anomaly import computeAnomalyScore
from nupic.encoders import MultiEncoder
from nupic.engine import Network


class DataBuffer(object):
    def __init__(self):
        self.stack = []

    def push(self, data):
        assert len(self.stack) == 0
        data = data.__class__(data)
        self.stack.append(data)

    def getNextRecordDict(self):
        assert len(self.stack) > 0
        return self.stack.pop()

class ClaClassifier():

    def __init__(self, net_structure, sensor_params, dest_region_params, class_encoder_params):

        self.run_number = 0

        # for classifier
        self.classifier_encoder_list = {}
        self.classifier_input_list   = {}
        self.prevPredictedColumns    = {}

        # TODO: 消したいパラメータ
        self.predict_value = class_encoder_params.keys()[0]
        self.predict_step  = 0


        # default param
        self.default_params = {
            'SP_PARAMS':  {
                "spVerbosity": 0,
                "spatialImp": "cpp",
                "globalInhibition": 1,
                "columnCount": 2024,
                "inputWidth": 0,             # set later
                "numActiveColumnsPerInhArea": 20,
                "seed": 1956,
                "potentialPct": 0.8,
                "synPermConnected": 0.1,
                "synPermActiveInc": 0.05,
                "synPermInactiveDec": 0.0005,
                "maxBoost": 2.0,
                },
            'TP_PARAMS': {
                "verbosity": 0,
                "columnCount": 2024,
                "cellsPerColumn": 32,
                "inputWidth": 2024,
                "seed": 1960,
                "temporalImp": "cpp",
                "newSynapseCount": 20,
                "maxSynapsesPerSegment": 32,
                "maxSegmentsPerCell": 128,
                "initialPerm": 0.21,
                "permanenceInc": 0.2,
                "permanenceDec": 0.1,
                "globalDecay": 0.0,
                "maxAge": 0,
                "minThreshold": 12,
                "activationThreshold": 16,
                "outputType": "normal",
                "pamLength": 1,
                },
            'CLASSIFIER_PARAMS':  {
                "clVerbosity": 0,
                "alpha": 0.005,
                "steps": "0"
                }
            }

        # tp
        self.tp_enable = True

        # net structure
        self.net_structure = OrderedDict()
        self.net_structure['sensor3'] = ['region1']
        self.net_structure['region1'] = ['region2']

        self.net_structure = net_structure

        # region change params
        self.dest_region_params = dest_region_params

        # sensor change params
        self.sensor_params = sensor_params

        self.class_encoder_params = class_encoder_params

        self._createNetwork()


    def _makeRegion(self, name, params):
        sp_name    = "sp_" + name
        if self.tp_enable:
            tp_name    = "tp_" + name
        class_name = "class_" + name

        # addRegion
        self.network.addRegion(sp_name, "py.SPRegion", json.dumps(params['SP_PARAMS']))
        if self.tp_enable:
            self.network.addRegion(tp_name, "py.TPRegion", json.dumps(params['TP_PARAMS']))
        self.network.addRegion( class_name, "py.CLAClassifierRegion", json.dumps(params['CLASSIFIER_PARAMS']))

        encoder = MultiEncoder()
        encoder.addMultipleEncoders(self.class_encoder_params)
        self.classifier_encoder_list[class_name]  = encoder
        if self.tp_enable:
            self.classifier_input_list[class_name]    = tp_name
        else:
            self.classifier_input_list[class_name]    = sp_name

    def _linkRegion(self, src_name, dest_name):
        sensor     =  src_name
        sp_name    = "sp_" + dest_name
        tp_name    = "tp_" + dest_name
        class_name = "class_" + dest_name

        if self.tp_enable:
            self.network.link(sensor, sp_name, "UniformLink", "")
            self.network.link(sp_name, tp_name, "UniformLink", "")
            self.network.link(tp_name, class_name, "UniformLink", "")
        else:
            self.network.link(sensor, sp_name, "UniformLink", "")
            self.network.link(sp_name, class_name, "UniformLink", "")


    def _initRegion(self, name):
        sp_name = "sp_"+ name
        tp_name = "tp_"+ name
        class_name = "class_"+ name

        # setting sp
        SP = self.network.regions[sp_name]
        SP.setParameter("learningMode", True)
        SP.setParameter("anomalyMode", True)

        # # setting tp
        if self.tp_enable:
            TP = self.network.regions[tp_name]
            TP.setParameter("topDownMode", False)
            TP.setParameter("learningMode", True)
            TP.setParameter("inferenceMode", True)
            TP.setParameter("anomalyMode", False)

        # classifier regionを定義.
        classifier = self.network.regions[class_name]
        classifier.setParameter('inferenceMode', True)
        classifier.setParameter('learningMode', True)


    def _createNetwork(self):

        def deepupdate(original, update):
            """
            Recursively update a dict.
            Subdict's won't be overwritten but also updated.
            """
            if update is None:
                return None
            for key, value in original.iteritems():
                if not key in update:
                    update[key] = value
                elif isinstance(value, dict):
                    deepupdate(value, update[key])
            return update


        self.network = Network()

        # check
        # if self.selectivity not in self.dest_region_params.keys():
        #     raise Exception, "There is no selected region : " + self.selectivity
        if not len(self.net_structure.keys()) == len(set(self.net_structure.keys())):
            raise Exception, "There is deplicated net_structure keys : " + self.net_structure.keys()

        # sensor
        for sensor_name, params in self.sensor_params.items():
            self.network.addRegion(sensor_name, "py.RecordSensor", json.dumps({"verbosity": 0}))
            sensor = self.network.regions[sensor_name].getSelf()

            # set encoder
            #params = deepupdate(cn.SENSOR_PARAMS, params)
            encoder = MultiEncoder()
            encoder.addMultipleEncoders( params )
            sensor.encoder         = encoder
            sensor.dataSource      = DataBuffer()


        # network
        print 'create element ...'
        for name in self.dest_region_params.keys():
            change_params = self.dest_region_params[name]
            params = deepupdate(self.default_params, change_params)
            # input width
            input_width = 0
            for source in [s for s,d in self.net_structure.items() if name in d]:
                if source in self.sensor_params.keys():
                    sensor = self.network.regions[source].getSelf()
                    input_width += sensor.encoder.getWidth()
                else:
                    input_width += params['TP_PARAMS']['cellsPerColumn'] * params['TP_PARAMS']['columnCount']

            params['SP_PARAMS']['inputWidth'] = input_width
            self._makeRegion(name, params)

        # link
        print 'link network ...'
        for source, dest_list in self.net_structure.items():
            for dest in dest_list:
                if source in self.sensor_params.keys():
                    self._linkRegion(source, dest)
                else:
                    if self.tp_enable:
                        self._linkRegion("tp_" + source, dest)
                    else:
                        self._linkRegion("sp_" + source, dest)

        # initialize
        print 'initializing network ...'
        self.network.initialize()
        for name in self.dest_region_params.keys():
            self._initRegion(name)

        return


    #@profile
    def run(self, input_data, learn=True, class_learn=True,learn_layer=None):
        """
        networkの実行.
        学習したいときは, learn=True, ftypeを指定する.
        予測したいときは, learn=False, ftypeはNoneを指定する.
        学習しているときも, 予測はしているがな.

        input_data = {'xy_value': [1.0, 2.0], 'ftype': 'sin'}
        """

        self.enable_learning_mode(learn, learn_layer)
        self.enable_class_learning_mode(class_learn)

        self.run_number += 1

        # calc encoder, SP, TP
        for sensor_name in self.sensor_params.keys():
            self.network.regions[sensor_name].getSelf().dataSource.push(input_data)
        self.network.run(1)
        #self.layer_output(input_data)
        #self.debug(input_data)


        # learn classifier
        inferences = {}
        for name in self.dest_region_params.keys():
            class_name = "class_" + name
            inferences['classifier_'+name]   = self._learn_classifier_multi(class_name, actValue=input_data[self.predict_value], pstep=self.predict_step)



        # anomaly
        #inferences["anomaly"] = self._calc_anomaly()

        return inferences


    def _learn_classifier_multi(self, region_name, actValue=None, pstep=0):
        """
        classifierの計算を行う.

        直接customComputeを呼び出さずに, network.runの中でやりたいところだけど,
        計算した内容の取り出し方法がわからない.
        """

        # TODO: networkとclassifierを完全に切り分けたいな.
        #       networkでは, sensor,sp,tpまで計算を行う.
        #       その計算結果の評価/利用は外に出す.

        classifier     = self.network.regions[region_name]
        encoder        = self.classifier_encoder_list[region_name].getEncoderList()[0]
        class_input    = self.classifier_input_list[region_name]
        tp_bottomUpOut = self.network.regions[class_input].getOutputData("bottomUpOut").nonzero()[0]
        #tp_bottomUpOut = self.network.regions["TP"].getSelf()._tfdr.infActiveState['t'].reshape(-1).nonzero()[0]

        if actValue is not None:
            bucketIdx = encoder.getBucketIndices(actValue)[0]
            classificationIn = {
                    'bucketIdx': bucketIdx,
                    'actValue': actValue
                    }
        else:
            classificationIn = {'bucketIdx': 0,'actValue': 'no'}
        clResults = classifier.getSelf().customCompute(
                recordNum=self.run_number,
                patternNZ=tp_bottomUpOut,
                classification=classificationIn
                )

        inferences= self._get_inferences(clResults, pstep, summary_tyep='sum')

        return inferences

    def _get_inferences(self, clResults, steps, summary_tyep='sum'):
        """
        classifierの計算結果を使いやすいように変更するだけ.
        """

        likelihoodsVec = clResults[steps]
        bucketValues   = clResults['actualValues']

        likelihoodsDict = defaultdict(int)
        bestActValue = None
        bestProb = None

        if summary_tyep == 'sum':
            for (actValue, prob) in zip(bucketValues, likelihoodsVec):
                likelihoodsDict[actValue] += prob
                if bestProb is None or likelihoodsDict[actValue] > bestProb:
                    bestProb = likelihoodsDict[actValue]
                    bestActValue = actValue

        elif summary_tyep == 'best':
            for (actValue, prob) in zip(bucketValues, likelihoodsVec):
                if bestProb is None or prob > bestProb:
                    likelihoodsDict[actValue] = prob
                    bestProb = prob
                    bestActValue = actValue

        return {'likelihoodsDict': likelihoodsDict, 'best': {'value': bestActValue, 'prob':bestProb}}


    def _calc_anomaly(self):
        """
        各層のanomalyを計算
        """

        score = 0
        anomalyScore = {}
        for name in self.dest_region_params.keys():
            #sp_bottomUpOut = self.network.regions["sp_"+name].getOutputData("bottomUpOut").nonzero()[0]
            sp_bottomUpOut = self.network.regions["tp_"+name].getInputData("bottomUpIn").nonzero()[0]

            if self.prevPredictedColumns.has_key(name):
                score = computeAnomalyScore(sp_bottomUpOut, self.prevPredictedColumns[name])
            #topdown_predict = self.network.regions["TP"].getSelf()._tfdr.topDownCompute().copy().nonzero()[0]
            topdown_predict = self.network.regions["tp_"+name].getSelf()._tfdr.topDownCompute().nonzero()[0]
            self.prevPredictedColumns[name] = copy.deepcopy(topdown_predict)

            anomalyScore[name] = score

        return anomalyScore

    def reset(self):
        """
        reset sequence
        """
        # for name in self.dest_region_params.keys():
        #     self.network.regions["tp_"+name].getSelf().resetSequenceStates()
        return

        # for sensor_name in self.sensor_params.keys():
        #     sensor = self.network.regions[sensor_name].getSelf()
        #     sensor.dataSource = DataBuffer()

    def enable_class_learning_mode(self, enable):
        for name in self.dest_region_params.keys():
            self.network.regions["class_"+name].setParameter("learningMode", enable)

    def enable_learning_mode(self, enable, layer_name = None):
        """
        各層のSP, TP, ClassifierのlearningModeを変更
        """
        if layer_name is None:
            for name in self.dest_region_params.keys():
                self.network.regions["sp_"+name].setParameter("learningMode", enable)
                if self.tp_enable:
                    self.network.regions["tp_"+name].setParameter("learningMode", enable)
                self.network.regions["class_"+name].setParameter("learningMode", enable)
        else:
            for name in self.dest_region_params.keys():
                self.network.regions["sp_"+name].setParameter("learningMode", not enable)
                if self.tp_enable:
                    self.network.regions["tp_"+name].setParameter("learningMode", not enable)
                self.network.regions["class_"+name].setParameter("learningMode", not enable)
            for name in layer_name:
                self.network.regions["sp_"+name].setParameter("learningMode", enable)
                if self.tp_enable:
                    self.network.regions["tp_"+name].setParameter("learningMode", enable)
                self.network.regions["class_"+name].setParameter("learningMode", enable)


    def print_inferences(self, input_data, inferences):
        """
        計算結果を出力する
        """

        # print "%10s, %10s, %1s" % (
        #         int(input_data['xy_value'][0]),
        #         int(input_data['xy_value'][1]),
        #         input_data['label'][:1]),
        print "%5s" % (
                input_data['label']),

        try:
            for name in sorted(self.dest_region_params.keys()):
                print "%5s" % (inferences['classifier_'+name]['best']['value']),

            for name in sorted(self.dest_region_params.keys()):
                print "%6.4f," % (inferences['classifier_'+name]['likelihoodsDict'][input_data[self.predict_value]]),
        except:
            pass

        # for name in sorted(self.dest_region_params.keys()):
        #     print "%3.2f," % (inferences["anomaly"][name]),

        # for name in sorted(self.dest_region_params.keys()):
        #     print "%5s," % name,

        print

    def layer_output(self, input_data, region_name=None):
        if region_name is not None:
            Region = self.network.regions[region_name]
            print Region.getOutputData("bottomUpOut").nonzero()[0]
            return

        for name in self.dest_region_params.keys():
            SPRegion = self.network.regions["sp_"+name]
            if self.tp_enable:
                TPRegion = self.network.regions["tp_"+name]

            print "#################################### ", name
            print
            print "==== SP layer ===="
            print "input:  ", SPRegion.getInputData("bottomUpIn").nonzero()[0][:20]
            print "output: ", SPRegion.getOutputData("bottomUpOut").nonzero()[0][:20]
            print
            if self.tp_enable:
                print "==== TP layer ===="
                print "input:  ", TPRegion.getInputData("bottomUpIn").nonzero()[0][:20]
                print "output: ", TPRegion.getOutputData("bottomUpOut").nonzero()[0][:20]
                print
            print "==== Predict ===="
            print TPRegion.getSelf()._tfdr.topDownCompute().copy().nonzero()[0][:20]
            print

    def save(self, path):
        import pickle
        with open(path, 'wb') as modelPickleFile:
            pickle.dump(self, modelPickleFile)




