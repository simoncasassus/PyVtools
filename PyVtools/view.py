#!/usr/bin/env python3
import sys
from astropy.io import fits 

import Vtools
from ImUtils.Cube2Im import slice0

filename_source=sys.argv[1]
print("loading filename",filename_source)

hdu = fits.open(filename_source)
if isinstance(hdu,list):
    hdu=hdu[0]

hdr = hdu.header
if (hdr['NAXIS'] > 2):
    print("Data is larger than 2D, keeping 1st plane")
    hdu=slice0(hdu)

if (len(sys.argv) > 2):
    filename_contours=sys.argv[2]
    hducontours = fits.open(filename_contours)
    if isinstance(hducontours,list):
        hducontours=hducontours[0]

    hdrcontours = hducontours.header
    if (hdrcontours['NAXIS'] > 2):
        print("Data is larger than 2D, keeping 1st plane")
        hducontours=slice0(hducontours)

    
    Vtools.View([hdu,hducontours])
else:
    Vtools.View(hdu)
