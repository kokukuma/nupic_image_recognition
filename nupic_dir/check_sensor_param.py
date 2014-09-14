#!/usr/bin/env python
import copy
import json
from nupic.algorithms.anomaly import computeAnomalyScore
from nupic.engine import Network
from nupic.encoders import MultiEncoder

from nupic_dir.lib.load_data import load_dataset, get_patch
from collections import defaultdict, Counter

train_data, train_label = load_dataset('./data/pylearn2_gcn_whitened/train.pkl')
patch_heigh = 3
patch_width = 3
patch_step  = 3


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
}


def createSensorEncoder():
    """Create the encoder instance for our test and return it."""
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
            "x": {
                "clipInput": True,
                "type": "VectorEncoderOPF",
                "dataType": "float",
                "n": 200,
                "w": 21,
                "length": 27,
                #"length": 192,
                "fieldname": u"x",
                "name": u"x",
                "maxval":  1.5,
                "minval": -1.5,
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

    network.initialize()


    # Make sure learning is enabled
    spatialPoolerRegion = network.regions["spatialPoolerRegion"]
    spatialPoolerRegion.setParameter("learningMode", True)
    spatialPoolerRegion.setParameter("anomalyMode", True)

    return network


def runNetwork(network):
    """
    """
    sensorRegion = network.regions["sensor"]
    SPRegion     = network.regions["spatialPoolerRegion"]

    prevPredictedColumns = []

    input_count  = Counter()
    sensor_count = Counter()
    sp_count     = Counter()

    for i, data in enumerate(train_data[:1000]):
        patch_data, movement = get_patch(data, height=patch_heigh, width=patch_width, step=patch_step)

        for patch in patch_data:
            input_len = reduce(lambda x,y: x * y, patch.shape)
            input_data = {
                    'x': patch.reshape((input_len)).tolist(),
                    }

            sensorRegion.getSelf().dataSource.push(input_data)
            network.run(1)

            for d in patch.reshape((input_len)).tolist():
                input_count[str(d)] += 1
            for d in SPRegion.getInputData("bottomUpIn").nonzero()[0]:
                sensor_count[str(d)]  += 1
            for d in SPRegion.getOutputData("bottomUpOut").nonzero()[0]:
                sp_count[str(d)]    += 1

            print len(input_count), len(sensor_count), sp_count


if __name__ == "__main__":
    network = createNetwork()

    runNetwork(network)
