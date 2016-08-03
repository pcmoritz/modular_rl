import unittest
import numpy as np
from numpy.testing import assert_equal

import ray
from modular_rl import ZFilter

def ray_assert_equal(a, b, attrs=[]):
    attrs.extend(["ray_deallocator", "ray_objectid"])
    a = a.__dict__
    b = b.__dict__
    A = dict((i, a[i]) for i in a if i not in attrs)
    B = dict((i, b[i]) for i in b if i not in attrs)
    assert_equal(A, B)

class SerializationTest(unittest.TestCase):

    def testSimple(self):
        ray.init(start_ray_local=True, num_workers=1)
        filt = ZFilter((2,2))
        filt(np.array([[0.0, 1.0], [2.0, 3.0]]))
        x = ray.put(filt)
        y = ray.get(x)
        ray_assert_equal(y, filt, ["rs"])
        ray_assert_equal(y.rs, filt.rs)

if __name__ == "__main__":
    unittest.main()
