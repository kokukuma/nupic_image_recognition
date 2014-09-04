#!/usr/bin/env python
import copy
import json
from nupic.algorithms.anomaly import computeAnomalyScore
from nupic.engine import Network
from nupic.encoders import MultiEncoder


def get_input_data():
    data = []
    data.append({'x': 0, 'y':'label_1'})
    data.append({'x': 1, 'y':'label_1'})
    data.append({'x': 2, 'y':'label_1'})
    data.append({'x': 3, 'y':'label_1'})
    data.append({'x': 4, 'y':'label_1'})
    data.append({'x': 5, 'y':'label_2'})
    data.append({'x': 6, 'y':'label_2'})
    data.append({'x': 7, 'y':'label_2'})
    data.append({'x': 8, 'y':'label_2'})
    data.append({'x': 9, 'y':'label_2'})
    return data

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

PARAMS = {
    'SP': {
        "spVerbosity": 0,
        "spatialImp": "cpp",
        "globalInhibition": 1,
        "columnCount": 2048,
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
                    "type": "ScalarEncoder",
                    "fieldname": u"x",
                    "name": u"x",
                    "maxval": 100.0,
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
                    "categoryList": ['label_1', 'label_2'],
                    "fieldname": u"y",
                    "name": u"y",
                    "w": 21,
            },
    })

    return encoder


def createNetwork():
    """Create the Network instance.

    The network has a sensor region reading data from `rataSource` and passing
    the encoded representation to an SPRegion. The SPRegion output is passed to
    a TPRegion.

    :param dataSource: a RecordStream instance to get data from
    :returns: a Network instance ready to run
    """
    network = Network()

    # Create Sensor
    network.addRegion("sensor", "py.RecordSensor", json.dumps({"verbosity": 0}))
    sensor = network.regions["sensor"].getSelf()
    sensor.encoder    = createSensorEncoder()
    sensor.dataSource = DataBuffer()

    # Add the spatial pooler region
    PARAMS['SP']["inputWidth"] = sensor.encoder.getWidth()
    network.addRegion("spatialPoolerRegion", "py.SPRegion", json.dumps(PARAMS['SP']))
    network.link("sensor", "spatialPoolerRegion", "UniformLink", "")

    # Add the TPRegion on top of the SPRegion
    network.addRegion("temporalPoolerRegion", "py.TPRegion", json.dumps(PARAMS['TP']))
    network.link("spatialPoolerRegion", "temporalPoolerRegion", "UniformLink", "")

    # Add classifier
    network.addRegion( "classifierRegion", "py.CLAClassifierRegion", json.dumps(PARAMS['CL']))

    network.initialize()



    # Make sure learning is enabled
    spatialPoolerRegion = network.regions["spatialPoolerRegion"]
    spatialPoolerRegion.setParameter("learningMode", True)
    spatialPoolerRegion.setParameter("anomalyMode", True)

    temporalPoolerRegion = network.regions["temporalPoolerRegion"]
    temporalPoolerRegion.setParameter("topDownMode", False)
    temporalPoolerRegion.setParameter("learningMode", True)
    temporalPoolerRegion.setParameter("inferenceMode", True)
    temporalPoolerRegion.setParameter("anomalyMode", False)

    classifierRegion = network.regions["classifierRegion"]
    classifierRegion.setParameter('inferenceMode', True)
    classifierRegion.setParameter('learningMode', True)

    return network


def runNetwork(network):
    """
    """
    sensorRegion = network.regions["sensor"]
    spatialPoolerRegion  = network.regions["spatialPoolerRegion"]
    temporalPoolerRegion = network.regions["temporalPoolerRegion"]
    classifierRegion     = network.regions["classifierRegion"]

    prevPredictedColumns = []

    for i, data in enumerate(get_input_data() * 100):
        # add data
        sensorRegion.getSelf().dataSource.push(data)

        # Run the network for a single iteration
        network.run(1)

        # Calculate the anomaly score using the active columns
        # and previous predicted columns
        activeColumns = spatialPoolerRegion.getOutputData("bottomUpOut").nonzero()[0]
        anomalyScore = computeAnomalyScore(activeColumns, prevPredictedColumns)
        prevPredictedColumns = copy.deepcopy(activeColumns)

        # Classifier
        activeCells = temporalPoolerRegion.getOutputData("bottomUpOut").nonzero()[0]
        res = classifierRegion.getSelf().customCompute(
                        recordNum=i,
                        patternNZ=activeCells,
                        classification={
                            'bucketIdx': createClassifierEncoder().getBucketIndices(data)[0] ,
                            'actValue': data['y']}
                        )
        predict = res['actualValues'][res[0].tolist().index(max(res[0]))]
        rate    = max(res[0])
        print '%s  x:%d  y:%s  p:%s  rate:%5.2f  anomaly:%5.2f' % (i, data['x'], data['y'], predict, rate, anomalyScore)




if __name__ == "__main__":
    network = createNetwork()

    runNetwork(network)
