"""
This module tests stacked_autoencoders.ipynb
"""

import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train_layer1(yaml_file_path, save_path, data_path):

    yaml = open("{0}/dae_l1.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 5000,
                    'batch_size': 50,
                    'monitoring_batches': 1,
                    'nhid': 100,
                    'max_epochs': 10,
                    'data_path': data_path,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_layer2(yaml_file_path, save_path, data_path):

    yaml = open("{0}/dae_l2.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 5000,
                    'batch_size': 50,
                    'monitoring_batches': 1,
                    'nvis': 100,
                    #'nvis': 3072,
                    'nhid': 100,
                    'max_epochs': 10,
                    'data_path': data_path,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_mlp(yaml_file_path, save_path, data_path):

    yaml = open("{0}/dae_mlp.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 5000,
                    'valid_stop': 55000,
                    'batch_size': 50,
                    'max_epochs': 10,
                    'data_path': data_path,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def test_sda():

    skip.skip_if_no_data()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  'yaml/stacked_autoencoders/'))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))

    train_layer1(yaml_file_path, save_path, data_path)
    train_layer2(yaml_file_path, save_path, data_path)
    train_mlp(yaml_file_path, save_path, data_path)

    try:
        os.remove("{}/dae_l1.pkl".format(save_path))
        os.remove("{}/dae_l2.pkl".format(save_path))
    except OSError:
        pass

if __name__ == '__main__':
    test_sda()
