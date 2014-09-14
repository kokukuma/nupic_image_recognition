#!/usr/bin/python
# coding: utf-8

import random
import numpy

class function_data(object):
    """
    指定した関数のx,yデータを取得する.

    sample:
        fd = function_data()
        ftype = fd.romdom_choice()
        data = fd.get_data(ftype)

        print ftype
        for d in data:
            print d
    """
    def __init__(self):
        """
        x = 0 - 100
        y = 0 - 100
        """
        self.max_x = 100
        self.function_label = {
                'flat':  ['flat'],
                'plus': ['plus'],
                'minus': ['minus'],
                }
        self.function_list = {
                'flat':  lambda x: 50.0,
                'plus':  lambda x: float(x),
                'minus': lambda x: 100-float(x),
                }

    def romdom_choice(self):
        ftype = random.choice(self.function_list.keys())
        return ftype

    def get_label(self, ftype):
        for label, ftypes in self.function_label.items():
            if ftype in ftypes:
                return label

    def get_data(self, ftype, error_var=0):
        """
        error=True の場合, 正規分布からサンプリングした誤差をのせる.
        """
        import numpy
        if ftype not in self.function_list.keys():
            return []
        result = []
        for x in range(self.max_x):
            if error_var == 0:
                y = self.function_list[ftype](x)
            else:
                y = self.function_list[ftype](x) + numpy.random.normal(0, error_var, 1)[0]
            result.append([float(x), y])
        return result


if __name__ == '__main__':
    import csv
    fd = function_data()
    with open('function_data.csv', 'wb') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(['x', 'y', 'function'])
        csvWriter.writerow([ 'float', 'float', 'string'])
        csvWriter.writerow(['', '', ''])

        for ftype in fd.function_list.keys():
            print ftype
            for d in fd.get_data(ftype):
                csvWriter.writerow([d[0], d[1], ftype])

