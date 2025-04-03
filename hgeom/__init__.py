# import sys
# import os
# sys.path.append(os.path.dirname(__file__))  # no clue why this is needed

__version__ = '2.0.6'

import sys
import os
import glob
from pathlib import Path

pkgroot = Path(__file__).resolve().parent

if glob.glob('_build/_bvh.cpy'):
    build = pkgroot / '_build'
    os.system('cd _build && ninja > /dev/null')
    sys.path.insert(0, str(build))  # Add the build path to sys.path for imports
    from _bvh import *
    from _cookie_cutter import *
    from _expand_xforms import *
    from _xform_dist import *
    from _bcc import *
    from _xbin import *
    from _xbin_util import *
    from _phmap import *
    from _qcp import *

    sys.path.pop(0)  # Remove the build path so it doesn't interfere with import
else:
    # from hgeom._bvh_nd import *
    from hgeom._bvh import *
    from hgeom._cookie_cutter import *
    from hgeom._expand_xforms import *
    from hgeom._xform_dist import *
    from hgeom._bcc import *
    from hgeom._xbin import *
    from hgeom._xbin_util import *
    from hgeom._phmap import *
    from hgeom._qcp import *

from hgeom import homog as homog
from hgeom.timer import Timer as Timer


def xform_dist2(*args):
    c, o = xform_dist2_split(*args)
    return c + o


BVH = SphereBVH_double
BVH32 = SphereBVH_float
BVH64 = SphereBVH_double

Xbin = Xbin_float
create_Xbin_nside = create_Xbin_nside_float
