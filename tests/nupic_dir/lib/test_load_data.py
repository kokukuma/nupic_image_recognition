#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest2 as unittest
import numpy
from nupic_dir.lib.load_data import get_patch

class load_data(unittest.TestCase):
    def setUp(self):
        self.height   = 3
        self.width    = 3
        self.channel  = 3
        self.tdata = numpy.zeros([ self.height, self.width, self.channel], dtype='float')

        """
        set ...
        0 0 0   0 0 0   0 0 0
        4 5 0,  5 6 0,  6 7 0
        7 8 0,  8 9 0  9 10 0
        """
        self.tdata[0][0] = numpy.asarray([7., 8., 9.] )
        self.tdata[0][1] = numpy.asarray([8., 9., 10.])
        self.tdata[1][0] = numpy.asarray([4., 5., 6.] )
        self.tdata[1][1] = numpy.asarray([5., 6., 7.] )


    def test_get_patch(self):
        # [1, 1]
        patch_data, movement = get_patch(self.tdata, 1, 1, 1)
        self.assertEqual(patch_data.shape, (9,1,1,3))
        self.assertEqual(patch_data.tolist()[0][0][0], [7., 8., 9.] )
        self.assertEqual(patch_data.tolist()[1][0][0], [8., 9., 10.])
        self.assertEqual(patch_data.tolist()[2][0][0], [0., 0., 0.] )
        self.assertEqual(patch_data.tolist()[3][0][0], [0., 0., 0.] )
        self.assertEqual(patch_data.tolist()[4][0][0], [5., 6., 7.] )

    def test_get_patch_form_2(self):
        # [2, 2]
        patch_data, movement = get_patch(self.tdata, 2, 2, 1)
        self.assertEqual(patch_data.shape, (4,2,2,3))
        self.assertEqual(patch_data.tolist()[0][0][0], [7., 8., 9.] )
        self.assertEqual(patch_data.tolist()[0][0][1], [8., 9., 10.])
        self.assertEqual(patch_data.tolist()[0][1][0], [4., 5., 6.] )
        self.assertEqual(patch_data.tolist()[0][1][1], [5., 6., 7.] )

        self.assertEqual(patch_data.tolist()[1][0][0], [8., 9., 10.] )
        self.assertEqual(patch_data.tolist()[1][0][1], [0., 0., 0.])
        self.assertEqual(patch_data.tolist()[1][1][0], [5., 6., 7.] )
        self.assertEqual(patch_data.tolist()[1][1][1], [0., 0., 0.] )

    def test_get_patch_step_2(self):
        # step = 2
        patch_data, movement = get_patch(self.tdata, 2, 2, 2)
        self.assertEqual(patch_data.shape, (1,2,2,3))
        self.assertEqual(patch_data.tolist()[0][0][0], [7., 8., 9.] )
        self.assertEqual(patch_data.tolist()[0][0][1], [8., 9., 10.])
        self.assertEqual(patch_data.tolist()[0][1][0], [4., 5., 6.] )
        self.assertEqual(patch_data.tolist()[0][1][1], [5., 6., 7.] )

if __name__ == '__main__':
    unittest.main()
