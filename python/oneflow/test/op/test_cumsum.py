"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import random
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest


def test_cumsum_dim(test_case, numpy_x, golden, dim):
    z = flow.tensor(numpy_x)
    z_cumsum = flow.cumsum(z, dim)
    test_case.assertTrue(np.allclose(golden, z_cumsum.numpy()))
    

@flow.unittest.skip_unless_1n1d()
class TestTensorIndexing(flow.unittest.TestCase):
    def test_basic_slice(test_case):
        numpy_x = np.arange(3*3*3*3).reshape(3,3,3,3).astype(float)
        golden0 = np.array([[[[  0.,   1.,   2.], [  3.,   4.,   5.], [  6.,   7.,   8.]], [[  9.,  10.,  11.], [ 12.,  13.,  14.], [ 15.,  16.,  17.]], [[ 18.,  19.,  20.], [ 21.,  22.,  23.], [ 24.,  25.,  26.]]], [[[ 27.,  29.,  31.], [ 33.,  35.,  37.], [ 39.,  41.,  43.]], [[ 45.,  47.,  49.], [ 51.,  53.,  55.], [ 57.,  59.,  61.]], [[ 63.,  65.,  67.], [ 69.,  71.,  73.], [ 75.,  77.,  79.]]], [[[ 81.,  84.,  87.], [ 90.,  93.,  96.], [ 99., 102., 105.]], [[108., 111., 114.], [117., 120., 123.], [126., 129., 132.]], [[135., 138., 141.], [144., 147., 150.], [153., 156., 159.]]]])
        golden1 = np.array([[[[  0.,   1.,   2.], [  3.,   4.,   5.], [  6.,   7.,   8.]], [[  9.,  11.,  13.], [ 15.,  17.,  19.], [ 21.,  23.,  25.]], [[ 27.,  30.,  33.], [ 36.,  39.,  42.], [ 45.,  48.,  51.]]], [[[ 27.,  28.,  29.], [ 30.,  31.,  32.], [ 33.,  34.,  35.]], [[ 63.,  65.,  67.], [ 69.,  71.,  73.], [ 75.,  77.,  79.]], [[108., 111., 114.], [117., 120., 123.], [126., 129., 132.]]], [[[ 54.,  55.,  56.], [ 57.,  58.,  59.], [ 60.,  61.,  62.]], [[117., 119., 121.], [123., 125., 127.], [129., 131., 133.]], [[189., 192., 195.], [198., 201., 204.], [207., 210., 213.]]]]) 
        golden2 = np.array([[[[  0.,   1.,   2.], [  3.,   5.,   7.], [  9.,  12.,  15.]], [[  9.,  10.,  11.], [ 21.,  23.,  25.], [ 36.,  39.,  42.]], [[ 18.,  19.,  20.], [ 39.,  41.,  43.], [ 63.,  66.,  69.]]], [[[ 27.,  28.,  29.], [ 57.,  59.,  61.], [ 90.,  93.,  96.]], [[ 36.,  37.,  38.], [ 75.,  77.,  79.], [117., 120., 123.]], [[ 45.,  46.,  47.], [ 93.,  95.,  97.], [144., 147., 150.]]], [[[ 54.,  55.,  56.], [111., 113., 115.], [171., 174., 177.]], [[ 63.,  64.,  65.], [129., 131., 133.], [198., 201., 204.]], [[ 72.,  73.,  74.], [147., 149., 151.], [225., 228., 231.]]]]) 
        golden3 = np.array([[[[  0.,   1.,   3.], [  3.,   7.,  12.], [  6.,  13.,  21.]], [[  9.,  19.,  30.], [ 12.,  25.,  39.], [ 15.,  31.,  48.]], [[ 18.,  37.,  57.], [ 21.,  43.,  66.], [ 24.,  49.,  75.]]], [[[ 27.,  55.,  84.], [ 30.,  61.,  93.], [ 33.,  67., 102.]], [[ 36.,  73., 111.], [ 39.,  79., 120.], [ 42.,  85., 129.]], [[ 45.,  91., 138.], [ 48.,  97., 147.], [ 51., 103., 156.]]], [[[ 54., 109., 165.], [ 57., 115., 174.], [ 60., 121., 183.]], [[ 63., 127., 192.], [ 66., 133., 201.], [ 69., 139., 210.]], [[ 72., 145., 219.], [ 75., 151., 228.], [ 78., 157., 237.]]]]) 
        test_cumsum_dim(test_case, numpy_x, golden0, 0) 
        test_cumsum_dim(test_case, numpy_x, golden1, 1)
        test_cumsum_dim(test_case, numpy_x, golden2, 2)
        test_cumsum_dim(test_case, numpy_x, golden3, 3)


if __name__ == "__main__":
    unittest.main()
