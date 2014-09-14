#!/usr/bin/python
# coding: utf-8

"""
model.pyで定義したmodelのオブジェクト.
ClaClassifier, NetworkLogic, RegionLogicで利用しやすいようにするため.
"""

import numpy
import json
from nupic.encoders import MultiEncoder
from nupic.engine import Network

from cla_logic.util import deepupdate, DataBuffer

class ModelObject(object):
    def __init(self, net_structure, sensor_params, region_params, default_params, define_classifier, deine_anomaly):
        self.net_structure = net_structure
        self.sensor_params = sensor_params
        self.region_params = region_params
        self.default_region_params = default_params
        self.define_classifier = define_classifier
        self.define_anomaly    = define_anomaly

        if not len(self.net_structure.keys()) == len(set(self.net_structure.keys())):
            raise Exception, "There is deplicated net_structure keys : " + self.net_structure.keys()

    def get_source_regions(self, name):
        return [s for s,d in self.net_structure.items() if name in d]:

    def get_net_struct(self):
        return self.net_structure.items()

    def get_sensor_region_name(self):
        return [x['sensor_name'] for x in self.sensor_params]

    def get_region_name(self):
        return sorted(self.region_params.keys())

    def get_sensor_param(self, name):
        return self.sensor_params[name]

    def get_region_param(self, name):
        params = self.region_params[name]
        if deault:
            params = deepupdate(self.default_region_params, params)
        return params


    def is_sensor(self, name):
        if name in self.get_sensor_name():
            return True
        return False

    def is_tp_enable(self, name):
        param = self.get_region_param(name)
        if param.has_key('TP_PARAMS'):
            return True
        return False

    def is_classifier_enable(self, name):
        if name in get_classifier_region_name():
            return True
        return False

    def get_output_name(self, name):
        if self.is_sensor(name):
            return name
        elif is_tp_enable(name):
            return 'tp_' + name
        else:
            return 'sp_' + name

    def get_classifier_region_name(self):

        return self.define_classifier.keys()

    def get_classifier_encoder(self, name):

        return self.define_classifier[name]
    def get_anomaly_region_name(self):
        return self.define_anomaly.keys()
    def get_predict_step(self, name):
        return self.define_classifier[name]['step']

    def get_predict_step(self, name):
        return self.define_classifier[name]['predict_value'



