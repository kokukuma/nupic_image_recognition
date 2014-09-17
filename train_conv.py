"""
"""
import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse

def main():
    skip.skip_if_no_data()

    # setting
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pylearn2/result'))
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pylearn2/yaml'))

    # set hyper parameter
    yaml = open("{0}/conv_cifar10_mono.yaml".format(yaml_path), 'r').read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'test_stop': 50,
                    'batch_size': 50,
                    'output_channels_h0': 5,
                    'output_channels_h1': 10,
                    'max_epochs': 10,
                    'data_path': data_path,
                    'save_path': save_path}



    yaml = yaml % (hyper_params)

    # train
    train = yaml_parse.load(yaml)
    train.main_loop()


if __name__ == '__main__':
    main()
