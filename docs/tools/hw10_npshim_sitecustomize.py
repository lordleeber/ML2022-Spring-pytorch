# imgaug 0.4.0 (2020) reads np.sctypes at import time; NumPy 2 removed it. Put it back.
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64],
                  'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                  'float': [np.float16, np.float32, np.float64, np.longdouble],
                  'complex': [np.complex64, np.complex128, np.clongdouble],
                  'others': [bool, object, bytes, str, np.void]}
