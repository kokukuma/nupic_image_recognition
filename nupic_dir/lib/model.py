#!/usr/bin/python
# coding: utf-8

net_structure = {
        'sensor1': ['region1'],
        #'region1': ['region2']
        }

"""
you need delete [:] at l.298
repos/nupic/build/release/lib/python2.7/site-packages/nupic/regions/RecordSensor.py

https://github.com/numenta/nupic/issues/727
"""
sensor_params = {
    'sensor1': {
        "pixel": {
            "clipInput": True,
            "type": "VectorEncoderOPF",
            "dataType": "float",
            "n": 200,
            "w": 21,
            #"length": 27,
            #"length": 192,
            "length": 3072,
            "fieldname": u"pixel",
            "name": u"pixel",
            "maxval":  1.5,
            "minval": -1.5,
        },
    },
}

# TPを入力しなかったら, SPのみのregionとする.
dest_resgion_data = {
    'region1': {
        'SP_PARAMS':{
            "columnCount": 2024,
            "numActiveColumnsPerInhArea": 40,
            },
        # 'TP_PARAMS':{
        #     "cellsPerColumn": 32,
        #     },

        # Noneを指定すると, それを使わない. default値も設定されない.
        'TP_PARAMS': None,
        },
    # 'region2': {
    #     'SP_PARAMS':{
    #         "inputWidth": 2024 * 32,
    #         "columnCount": 2024,
    #         "potentialRadius": 100,   # default value: 16
    #         "numActiveColumnsPerInhArea": 40,
    #         },
    #     'TP_PARAMS':{
    #         "cellsPerColumn": 32,
    #         },
    #     },
}


define_classifier = {
    'region1': {
        'step': 0,
        'predict_value': 'label',
        'encoder': {
            "label": {
                "type": "CategoryEncoder",
                "fieldname": u"label",
                "name": u"label",
                "categoryList": [i for i in range(10)],
                "w": 21,
            }
        }
    }
}

define_anomaly = {
    'region1': {}
}

