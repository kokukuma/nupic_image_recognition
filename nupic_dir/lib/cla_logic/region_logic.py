#!/usr/bin/python
# coding: utf-8

"""
Regionで利用するロジック.
self.networkはここからしか呼ばない.
"""

import numpy
import json
from nupic.encoders import MultiEncoder
from nupic.engine import Network

from cla_logic.util import deepupdate, DataBuffer

class RegionLogic(object):

    def _makeSensorRegion(self, name, params):
        self.network.addRegion(name, "py.RecordSensor", json.dumps({"verbosity": 0}))
        sensor = self.network.regions[name].getSelf()

        # set encoder
        #params = deepupdate(cn.SENSOR_PARAMS, params)
        encoder = MultiEncoder()
        encoder.addMultipleEncoders( params )
        sensor.encoder         = encoder
        sensor.dataSource      = DataBuffer()


    def _makeRegion(self, name, params):
        self.network.addRegion("sp_"+name, "py.SPRegion", json.dumps(params['SP_PARAMS']))

        if self.model_obj.is_tp_enable(name):
            self.network.addRegion("tp_"+name, "py.TPRegion", json.dumps(params['TP_PARAMS']))

        if self.model_obj.is_classifier_enable(name):
            self.network.addRegion( "class_"+name, "py.CLAClassifierRegion", json.dumps(params['CLASSIFIER_PARAMS']))

            classifier_params = self.model_obj.get_classifier_encoder(name)
            encoder = MultiEncoder()
            encoder.addMultipleEncoders(classifier_params)
            self.classifier_encoder_list["class_"+name] = encoder
            self.classifier_input_list["class_"+name]   = self.model_obj.get_output_name(name)



    def _linkRegion(self, src_name, dest_name):
        self.network.link(src_name, "sp_"+dest_name, "UniformLink", "")

        if self.model_obj.is_tp_enable(dest_name):
            self.network.link("sp_"+dest_name, "tp_"+dest_name, "UniformLink", "")
            self.network.link("tp_"+dest_name, "class_" + dest_name, "UniformLink", "")
        else:
            self.network.link("sp_"+dest_name, "class_" + dest_name, "UniformLink", "")

    def _initNetwork(self):
        self.network.initialize()
        for name in self.model_obj.get_region_name():
            self._initRegion(name)

    def _initRegion(self, name):
        sp_name = "sp_"+ name
        tp_name = "tp_"+ name
        class_name = "class_"+ name

        # setting sp
        SP = self.network.regions[sp_name]
        SP.setParameter("learningMode", True)
        SP.setParameter("anomalyMode", True)

        # # setting tp
        if self.model_obj.is_tp_enable(name):
            TP = self.network.regions[tp_name]
            TP.setParameter("topDownMode", False)
            TP.setParameter("learningMode", True)
            TP.setParameter("inferenceMode", True)
            TP.setParameter("anomalyMode", False)

        # classifier regionを定義.
        classifier = self.network.regions[class_name]
        classifier.setParameter('inferenceMode', True)
        classifier.setParameter('learningMode', True)


    def _get_input_width(self, name):
        if self.model_obj.is_sensor(name):
            sensor = self.network.regions[source].getSelf()
            return sensor.encoder.getWidth()

        elif is_tp_enable(name):
            params = self.model_obj.get_region_param(name)
            return params['TP_PARAMS']['cellsPerColumn'] * params['TP_PARAMS']['columnCount']

        else:
            params = self.model_obj.get_region_param(name)
            return params['SP_PARAMS']['columnCount']


    def _reset(self, name):
        if self.model_obj.is_tp_enable(name)
            self.network.regions["tp_"+name].getSelf().resetSequenceStates()

    def _enable_learning_mode(self, name, enable):
        self.network.regions["sp_"+name].setParameter("learningMode", enable)
        self.network.regions["class_"+name].setParameter("learningMode", enable)

        if self.model_obj.is_tp_enable(name)
            self.network.regions["tp_"+name].setParameter("learningMode", enable)


