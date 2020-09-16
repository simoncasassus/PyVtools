#!/Users/simon/opt/anaconda3/bin/python
import sys
import os
import os.path
import numpy as np
import astropy
from astropy.io import fits 

from pprint  import pprint as pp

filename_source=sys.argv[1]
print("loading filename",filename_source)

hdu = fits.open(filename_source)

if (sys.argv[2:]):
    selected_keywords=sys.argv[2:]
    if isinstance(hdu,list):
        hdr0=hdu[0].header
    else:
        hdr0=hdu.header
    for akey in selected_keywords:
        print("%s: %s" % (akey, hdr0[akey]))
else:
    if isinstance(hdu,list):
        for ahdu in hdu:
            print("HDU "+str(ahdu))
            pp(ahdu.header)
    else:
        hdr0=hdu.header
        pp(hdr0)

