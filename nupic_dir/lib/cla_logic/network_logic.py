#!/usr/bin/python
# coding: utf-8

"""
Network で利用するロジック.
self.networkはここからしか呼ばない.

しかし, これ, Regionのロジックと分離する必要あるのか?

そして, 何をcla_classifierに書いて, 何をこっち側に書くのかはっきりとしていない.
"""

import numpy
import json
from nupic.encoders import MultiEncoder
from nupic.engine import Network

from cla_logic.util import deepupdate, DataBuffer

class NetworkLogic(object):
    def _run_network(self):
        self.network.run(1)

    def _input_sensor(self, sensor_name, input_data):
        self.network.regions[sensor_name].getSelf().dataSource.push(input_data)

    def _learn_classifier_multi(self, region_name, actValue=None, pstep=0):
        """
        TODO: network自前実装時に, これは network側に持たせる.
              一旦, network_modelでもいいかな.

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
        TODO: network自前実装時に, これは network側に持たせる.
              一旦, network_modelでもいいかな.

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

    def _calc_anomaly(self, name):
        """
        TODO: network自前実装時に, これは network側に持たせる.
              一旦, network_modelでもいいかな.

        各層のanomalyを計算
        """

        score = 0

        outname = self.model_obj.get_output_name(name)
        sp_bottomUpOut = self.network.regions[outname].getInputData("bottomUpIn").nonzero()[0]

        if self.prevPredictedColumns.has_key(name):
            score = computeAnomalyScore(sp_bottomUpOut, self.prevPredictedColumns[name])
        topdown_predict = self.network.regions[outname].getSelf()._tfdr.topDownCompute().nonzero()[0]
        self.prevPredictedColumns[name] = copy.deepcopy(topdown_predict)

        return score

