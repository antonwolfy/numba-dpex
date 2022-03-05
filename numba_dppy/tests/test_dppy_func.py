# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dpctl
import numpy as np

import numba_dppy
from numba_dppy.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestDPPYFunc:
    N = 257

    def test_dppy_func_device_array(self):
        @numba_dppy.func
        def g(a):
            return a + 1

        @numba_dppy.kernel
        def f(a, b):
            i = numba_dppy.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, numba_dppy.DEFAULT_LOCAL_SIZE](a, b)

        assert np.all(b == 2)

    def test_dppy_func_ndarray(self):
        @numba_dppy.func
        def g(a):
            return a + 1

        @numba_dppy.kernel
        def f(a, b):
            i = numba_dppy.get_global_id(0)
            b[i] = g(a[i])

        @numba_dppy.kernel
        def h(a, b):
            i = numba_dppy.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, numba_dppy.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 2)

            h[self.N, numba_dppy.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 3)
