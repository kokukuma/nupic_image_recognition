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

from cla_logic.util import deepupdate, DataBuffer
from cla_logic.result import Result
from cla_logic.region_logic import RegionLogic
from cla_logic.network_logic import NetworkLogic
from cla_logic.model_obj import ModelObject



class ClaClassifier(RegionLogic, NetworkLogic, Result):

    def __init__(self, net_structure, sensor_params, dest_region_params, define_classifier):

        self.run_number = 0

        # for classifier
        self.classifier_encoder_list = {}
        self.classifier_input_list   = {}
        self.prevPredictedColumns    = {}

        # TODO: 消したいパラメータ
        self.predict_value = define_classifier.keys()[0]
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

        # defind network
        self.network = Network()

        # network model
        self.network_model = NetworkModel()

        # region model
        self.model_obj = RegionModel(net_structure, sensor_params, dest_region_params, self.default_params, define_classifier)
        self.model = ModelObject(net_structure, sensor_params, dest_region_params, self.default_params)

        # sensor change params
        #self.sensor_params = sensor_params

        self.createNetwork()


    def createNetwork(self):

        # sensor
        for name in self.model_obj.get_sensor_name():
            params = self.model_obj.get_param(name)
            self._makeSensorRegion(name, params)

        # network
        print 'create element ...'
        for name in self.model_obj.get_region_name():
            params = self.model_obj.get_region_param(name)

            # input width
            input_width = 0
            for source in self.model_obj.get_source_regions(name):
                input_width += self._get_input_width(name)
            params['SP_PARAMS']['inputWidth'] = input_width
            self._makeRegion(name, params)

        # link
        print 'link network ...'
        for source, dest_list in self.model_obj.get_net_struct():
            src_name = self.model_obj.get_output_name(source)
            for dest in dest_list:
                self._linkRegion(src_name, dest)


        # initialize
        print 'initializing network ...'
        self._initNetwork()

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

        self.run_number += 1

        # calc encoder, SP, TP
        for sensor_name in get_region_name()
            self._input_sensor(sensor_name, input_data)
        self._run_network()

        # learn classifier
        inferences = {}
        for name in self.model_obj.get_classifier_region_name():
            predict_value  = self.model_obj.get_predict_value(name)
            predict_step   = self.model_obj.get_predict_step(name)
            inferences['classifier_' + name] = \
                    self._learn_classifier_multi( "class_" + name,
                                                  actValue=input_data[predict_value],
                                                  pstep=predict_step)

        # anomaly
        for name in self.model_obj.get_anomaly_region_name():
            inferences["anomaly"+name] = self._calc_anomaly(name)

        return inferences


    def reset(self):
        """
        reset sequence
        """
        for name in self.model_obj.get_region_name():
            self._reset(name)

    def enable_learning_mode(self, enable, layer_name = None):
        """
        各層のSP, TP, ClassifierのlearningModeを変更
        """
        if layer_name is None:
            for name in self.model_obj.get_region_name():
                self._enable_learning_mode(name, enable)
        else:
            for name in self.model_obj.get_region_name():
                self._enable_learning_mode(name, not enable)
            for name in layer_name:
                self._enable_learning_mode(name, enable)

    def save(self, path):
        import pickle
        with open(path, 'wb') as modelPickleFile:
            pickle.dump(self, modelPickleFile)




