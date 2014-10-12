#!/usr/bin/env python
# encoding: utf-8
import numpy
import copy
import json
from collections import defaultdict

#from nupic.research.spatial_pooler import SpatialPooler as SP
#from nupic.algorithms.CLAClassifier import CLAClassifier as CL
from nupic.bindings.algorithms import SpatialPooler as SP
from nupic.research.TP10X2 import TP10X2 as TP
from nupic.bindings.algorithms import FastCLAClassifier as CL


class Network(object):
    def __init__(self):
        self.regions = {}
        self.network = defaultdict(list)

        # TODO: これは良くない.
        self.calc_sort = []

    def addRegion(self, name, type, params):
        if type == "SP":
            self.regions[name] = Region("SP", SP(**params))
        elif type == "TP":
            self.regions[name] = Region("TP", TP(**params))
        elif type == "CL":
            self.regions[name] = Region("CL", CL(**params))

    def link(self, src_region, dst_region):
        """
        計算する順番に入力
        """
        # TODO: これは良くない.
        self.calc_sort.append(src_region)
        if dst_region not in self.calc_sort:
            self.calc_sort.append(dst_region)

        self.network[src_region].append(dst_region)

    def initialize(self):
        """
        計算する順番を確定する.

        TODO: 下記構造にも対応できるようにする.
              + 入力が複数ある場合
              + 入力が下記regionの合成になる場合.

        regino1 -+- region3
                 |
        region2 -+

        TODO: network-graph obj を作った方が綺麗になりそう.
          + 指定したノードの前に繋がっているノード出力
          + 計算すべき順番にノード名をlistで返す.
        """
        pass

    #@profile
    def run(self, inputArray):
        """
        inputArray = np.zeros(self.inputSize)

        基本的に１次元, SPの場合のみ, SPの次元に変換して入出力する.

        TODO: initializeのTODOにある構造に対応するため,
              入力を単純に前回の計算結果ではなく,
              その層と繋がっている前の層の出力を得るようにする.
        """

        for region in self.sortedRegion():

            if region.type == "SP":
                idim = region.getInputDim()
                odim = region.getOutputDim()
                region.outputArray = numpy.zeros( [reduce(lambda x,y: x*y, odim)], dtype="float32" )

                # self.activeArray = np.zeros(self.columnNumber)
                region.getObj().compute(
                        inputArray.reshape([reduce(lambda x,y: x*y, idim)]),
                        region.getLearnmode(),
                        region.outputArray)
                #inputArray = outputArray.reshape(tuple(reduce(lambda x,y: x*y, odim)))
                inputArray = region.outputArray

            elif region.type == "TP":
                region.outputArray = region.getObj().compute(
                        inputArray.astype("float32"),
                        enableLearn=region.getLearnmode(),
                        computeInfOutput = True)
                inputArray = region.outputArray

        return region.outputArray

    def sortedRegion(self):
        return [self.regions[x] for x in self.calc_sort]

    def getOutputByRegion(self, name, strategy="union"):
        """
        指定したRegionの出力を返す.
        複数指定したらそれらを結合/合成した結果を返す.
        """
        pass

    def reset(self):
        for region in self.sortedRegion():
            if region.type == "TP":
                region.getObj().reset()


class Region(object):
    def __init__(self, type, obj):
        self.type       = type
        self.obj        = obj
        self.learnmode  = True
        self.outputArray = None

    def getObj(self):
        return self.obj

    def getInput(self):
        pass

    def getOutput(self):
        return self.outputArray

    def getPredictColumn(self):
        if self.type == "TP":
            return self.getObj().topDownCompute()

    def getInputDim(self):
        if self.type == "SP":
            return self.getObj().getInputDimensions()

        elif self.type == "TP":
            pass

    def getOutputDim(self):
        if self.type == "SP":
            return self.getObj().getColumnDimensions()

        elif self.type == "TP":
            pass

    def getInputSize(self):
        if self.type == "SP":
            return self.getObj().getNumInputs()

        elif self.type == "TP":
            return self.getObj().numberOfCols

    def getOutputSize(self):
        if self.type == "SP":
            return self.getObj().getNumColumns()

        elif self.type == "TP":
            return self.getObj().getNumCells()


    def getLearnmode(self):
        return self.learnmode

    def setLearnmode(self, enable):
        self.learnmode = enable
