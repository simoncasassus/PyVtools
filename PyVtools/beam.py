#!/usr/bin/env python3
import sys
from astropy.io import fits 

filename_source=sys.argv[1]
print(filename_source)
hdu = fits.open(filename_source)

hdr = hdu[0].header

if 'BMAJ' in hdr.keys():
    bmaj=hdr['BMAJ']*3600.
    bmin=hdr['BMIN']*3600.
    bpa=hdr['BPA']
elif (len(hdu)>1):
    print("no beam info, look for extra HDU")
    beamhdr=hdu[1].header

    #print("beamhdr")
    #from pprint import pprint
    #pprint(beamhdr)
    
    beamdata=hdu[1].data
    #print("beamvector", beamdata)

    bmaj=beamdata[0][0]
    bmin=beamdata[0][1]
    bpa=beamdata[0][2]
    
    print("beam vector %.3f %.3f %.3f  ---> %.3f %.3f %.3f  " % (beamdata[0][0],beamdata[0][1],beamdata[0][2],beamdata[-1][0],beamdata[-1][1],beamdata[1][2]))

    
print( "%.3f x %.3f / %.1f " % (bmaj,bmin,bpa))
    

