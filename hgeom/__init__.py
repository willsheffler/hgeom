# import sys
# import os
# sys.path.append(os.path.dirname(__file__))  # no clue why this is needed

# from ._bvh_nd import *
from ._bvh import *
from ._cookie_cutter import *
from ._expand_xforms import *
from ._xform_dist import *

from ._bcc import *
from ._xbin import *
from ._xbin_util import *

from ._phmap import *
from ._qcp import *

# sys.path.pop()

from . import homog
from .timer import Timer

def xform_dist2(*args):
    c, o = xform_dist2_split(*args)
    return c + o

BVH = SphereBVH_double
BVH32 = SphereBVH_float
BVH64 = SphereBVH_double

Xbin = Xbin_float
create_Xbin_nside = create_Xbin_nside_float
