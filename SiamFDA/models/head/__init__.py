from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SiamFDA.models.head.fda import DepthwiseFDAM, MultiFDAM


FDAM = {
        'DepthwiseFDAM': DepthwiseFDAM,
        'MultiFDAM': MultiFDAM
       }


def get_fdam_head(name, **kwargs):
    return FDAM[name](**kwargs)

