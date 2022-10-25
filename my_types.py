from typing import Dict, Tuple, Union,Callable, List, NewType,Any,Set
from scipy import sparse



MATRIX = NewType('MATRIX',sparse.spmatrix)

TOPO_TPYE = NewType('TOPO_TPYE',str)

OUTPUT_PATH = NewType('OUTPUT_PATH',str) 
RTREE_PATH = NewType('RTREE_PATH',str)
RTREE_CC_PATH = NewType('RTREE_CC_PATH',str)
CC2AS_PATH = NewType('CC2AS_PATH',str)
CC_PATH = NewType('CC2AS_PATH',str)
WEIGHT_PATH = NewType('WEIGHT_PATH',str)

COUNTRY_CODE = NewType('COUNTRY_CODE',str)
AS_CODE = NewType('AS_CODE', int or str)