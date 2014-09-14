#!/usr/bin/python
# coding: utf-8

import copy

class Result(object):
    def __init__(self):
        pass

    def print_network(self):
        """
        network構造を出力する
        """
        pass

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
            for name in self.region_model.get_region_name():
                print "%5s" % (inferences['classifier_'+name]['best']['value']),

            for name in self.region_model.get_region_name():
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

        for name in self.region_model.get_region_name():
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
