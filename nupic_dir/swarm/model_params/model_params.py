MODEL_PARAMS = \
{ 'aggregationInfo': { 'days': 0,
                       'fields': [],
                       'hours': 0,
                       'microseconds': 0,
                       'milliseconds': 0,
                       'minutes': 0,
                       'months': 0,
                       'seconds': 0,
                       'weeks': 0,
                       'years': 0},
  'model': 'CLA',
  'modelParams': { 'anomalyParams': { u'anomalyCacheRecords': None,
                                      u'autoDetectThreshold': None,
                                      u'autoDetectWaitRecords': None},
                   'clParams': { 'alpha': 0.08751250000000001,
                                 'clVerbosity': 0,
                                 'regionName': 'CLAClassifierRegion',
                                 'steps': '0'},
                   'inferenceType': 'NontemporalClassification',
                   'sensorParams': { 'encoders': { '_classifierInput': { 'classifierOnly': True,
                                                                         'clipInput': True,
                                                                         'fieldname': 'function_type',
                                                                         'n': 90,
                                                                         'name': '_classifierInput',
                                                                         'type': 'AdaptiveScalarEncoder',
                                                                         'w': 21},
                                                   u'y_value': { 'clipInput': True,
                                                                 'fieldname': 'y_value',
                                                                 'maxval': 100.0,
                                                                 'minval': 0.0,
                                                                 'n': 334,
                                                                 'name': 'y_value',
                                                                 'type': 'ScalarEncoder',
                                                                 'w': 21}},
                                     'sensorAutoReset': None,
                                     'verbosity': 0},
                   'spEnable': False,
                   'spParams': { 'columnCount': 2048,
                                 'globalInhibition': 1,
                                 'inputWidth': 0,
                                 'maxBoost': 2.0,
                                 'numActiveColumnsPerInhArea': 40,
                                 'potentialPct': 0.8,
                                 'seed': 1956,
                                 'spVerbosity': 0,
                                 'spatialImp': 'cpp',
                                 'synPermActiveInc': 0.05,
                                 'synPermConnected': 0.1,
                                 'synPermInactiveDec': 0.0005},
                   'tpEnable': False,
                   'tpParams': { 'activationThreshold': 16,
                                 'cellsPerColumn': 32,
                                 'columnCount': 2048,
                                 'globalDecay': 0.0,
                                 'initialPerm': 0.21,
                                 'inputWidth': 2048,
                                 'maxAge': 0,
                                 'maxSegmentsPerCell': 128,
                                 'maxSynapsesPerSegment': 32,
                                 'minThreshold': 12,
                                 'newSynapseCount': 20,
                                 'outputType': 'normal',
                                 'pamLength': 1,
                                 'permanenceDec': 0.1,
                                 'permanenceInc': 0.1,
                                 'seed': 1960,
                                 'temporalImp': 'cpp',
                                 'verbosity': 0},
                   'trainSPNetOnlyIfRequested': False},
  'predictAheadTime': None,
  'version': 1}
