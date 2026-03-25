import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint as pp
from copy import deepcopy
import sys

if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

#include_path = os.environ['HOME'] + '/common/python/include/'
#sys.path.append(include_path)
from ImUtils.Resamp import gridding

icmap = 0

cmaps = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
    'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
    'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
    'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
    'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
    'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
    'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
    'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
    'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
    'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
    'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
    'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
    'winter_r'
]

## WIDGET INBUILT KEY STROKES
# g: grid

# https://matplotlib.org/users/event_handling.html
# https://matplotlib.org/3.1.1/gallery/widgets/rectangle_selector.html






class VtoolsViewer:
    def __init__(self, indata, cmap='RdBu_r', AllContours=False,cmapcontours='Greens_r', contlevels=[0.2, 0.4, 0.6, 0.8]):
        self.hdu = None
        self.subim_max = 0.
        self.icmap = 0
        self.selected_subim = None
        self.selected_rms = None
        self.selected_i1m = None
        self.selected_j1m = None
        self.selected_i2m = None
        self.selected_j2m = None
        self.selected_range1 = None
        self.selected_range2 = None
        self.last_fit_model = None
        self.last_fit_ellipse = None
        self.last_fit_is_subregion = False
        self.fit_setup_stage = None
        self.interactive_init_phys = {}
        self.event_handler_id = None
        self.motion_event_handler_id = None
        self.fit_artists = []
        self.xlim_store = None
        self.ylim_store = None
        self.xlim_cid = None
        self.ylim_cid = None

        AddContours = False
        if isinstance(indata, list) or isinstance(indata, tuple):
            self.hdu = indata[0]
            if isinstance(self.hdu, astropy.io.fits.hdu.hdulist.HDUList):
                self.hdu = self.hdu[0]
            elif isinstance(indata, np.ndarray):
                self.hdu = fits.PrimaryHDU()
                self.hdu.data = indata
                hdr = self.hdu.header
                hdr['CDELT1'] = 1.
                hdr['CDELT2'] = 1.
                (nx, ny) = indata.shape
                hdr['CRPIX1'] = int(nx / 2.)
                hdr['CRPIX2'] = int(ny / 2.)
                hdr['CTYPE1'] = 'pixel'
                hdr['CTYPE2'] = 'pixel'

            if (len(indata) > 1):
                hducontours = indata[1]
                AddContours = True
                if isinstance(hducontours, astropy.io.fits.hdu.hdulist.HDUList):
                    hducontours = hducontours[0]
                elif isinstance(hducontours, np.ndarray):
                    imcont = indata[1]
                    hducontours = fits.PrimaryHDU()
                    hducontours.data = imcont
                    hdr = hducontours.header
                    hdrcontours = {}
                    hdrcontours['CDELT1'] = 1.
                    hdrcontours['CDELT2'] = 1.
                    (nx, ny) = imcont.shape
                    hdrcontours['CRPIX1'] = int(nx / 2.)
                    hdrcontours['CRPIX2'] = int(ny / 2.)
                    hdrcontours['CTYPE1'] = 'pixel'
                    hdrcontours['CTYPE2'] = 'pixel'
        else:
            self.hdu = indata
            if isinstance(self.hdu, astropy.io.fits.hdu.hdulist.HDUList):
                self.hdu = self.hdu[0]
            elif isinstance(indata, np.ndarray):
                self.hdu = fits.PrimaryHDU()
                self.hdu.data = indata
                hdr = self.hdu.header
                hdr['CDELT1'] = 1.
                hdr['CDELT2'] = 1.
                (nx, ny) = indata.shape
                hdr['CRPIX1'] = int(nx / 2.)
                hdr['CRPIX2'] = int(ny / 2.)
                hdr['CTYPE1'] = 'pixel'
                hdr['CTYPE2'] = 'pixel'
        
        im = np.squeeze(self.hdu.data)
        hdr = self.hdu.header

        if (not 'CTYPE1' in hdr.keys()):
            hdr['CTYPE1'] = 'pixel'
            hdr['CTYPE2'] = 'pixel'
            (nx, ny) = im.shape
            hdr['CRPIX1'] = int(nx / 2.)
            hdr['CRPIX2'] = int(ny / 2.)

        self.beam = False
        import re
        if 'BUNIT' in hdr.keys():
            if re.search('beam', hdr['BUNIT'], re.IGNORECASE):
                self.beam = (np.pi / (4. * np.log(2.))) * (hdr['BMAJ'] * hdr['BMIN'] /
                                                      (hdr['CDELT2']**2))

        plt.close('all')
        self.fig, self.ax = plt.subplots()

        ctype1 = hdr.get('CTYPE1', 'X')
        ctype2 = hdr.get('CTYPE2', 'Y')
        self.ax.set_label(f'{ctype2} vs {ctype1}')

        if 'f' in plt.rcParams['keymap.fullscreen']:
            plt.rcParams['keymap.fullscreen'].remove('f')

        (d0, a0) = self.pix2wcs_0CRVAL(0., 0.)
        (d1, a1) = self.pix2wcs_0CRVAL(hdr['NAXIS2'] - 1, hdr['NAXIS1'] - 1)

        range2 = np.max(im[np.where(np.isfinite(im))])
        range1 = np.min(im[np.where(np.isfinite(im))])

        self.theimage = self.ax.imshow(
            im,
            origin='lower',
            cmap=cmap,
            extent=[a0, a1, d0, d1],
            vmin=range1,
            vmax=range2,
            interpolation='nearest')

        self._colorbar(self.theimage)
        self.ax.set_aspect('equal', adjustable='box')
        self.update_selection_from_view()

        if AllContours:
            levels = np.array(contlevels) * np.max(im)
            self.ax.contour(im,
                        origin='lower',
                        extent=[a0, a1, d0, d1],
                        levels=levels,
                        cmap=cmap)

        if AddContours:
            hducontours_matched = gridding(hducontours, hdr, ReturnHDU=True)
            imcontours = hducontours_matched.data
            levels = np.array([0.2, 0.4, 0.6, 0.8]) * np.max(imcontours)
            if AllContours:
                self.ax.contour(imcontours,
                            origin='lower',
                            colors='green',
                            extent=[a0, a1, d0, d1],
                            levels=levels,
                            linewidths=2.0,
                            alpha=0.5)
            else:
                self.ax.contour(imcontours,
                            origin='lower',
                            cmap=cmapcontours,
                            extent=[a0, a1, d0, d1],
                            levels=levels)

        self.rs = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=False,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True)
        self.rs.set_active(False)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.xlim_cid = self.ax.callbacks.connect('xlim_changed', self.on_view_changed)
        self.ylim_cid = self.ax.callbacks.connect('ylim_changed', self.on_view_changed)

    def on_view_changed(self, ax):
        self.update_selection_from_view()

    def _colorbar(self, Mappable, Orientation='vertical'):
        Ax = Mappable.axes
        fig = Ax.figure
        divider = make_axes_locatable(Ax)
        Cax = divider.append_axes("right", size="5%", pad=0.08)
        return fig.colorbar(mappable=Mappable,
                            cax=Cax,
                            use_gridspec=True,
                            orientation=Orientation,
                            format="%.1e")

    def pix2wcs_0CRVAL(self, j, i):
        hdr = self.hdu.header
        if ('pixel' in hdr['CTYPE1']):
            pixscale = 1.
        else:
            pixscale = 3600. * hdr['CDELT2']
        a = -pixscale * (i - (hdr['CRPIX1'] - 1.))
        d = pixscale * (j - (hdr['CRPIX2'] - 1.))
        return (d, a)

    def wcs2pix_0CRVAL(self, d, a):
        hdr = self.hdu.header
        if ('pixel' in hdr['CTYPE1']):
            pixscale = 1.
        else:
            pixscale = 3600. * hdr['CDELT2']
        i = -a / pixscale + (hdr['CRPIX1'] - 1.)
        j = d / pixscale + (hdr['CRPIX2'] - 1.)
        return (j, i)

    def update_selection_from_view(self):
        """Updates the selected region to match the current axes limits."""
        print("Updating region to current view.")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        im = np.squeeze(self.hdu.data)
        hdr = self.hdu.header

        # Convert view limits (in data coords/arcsec) to pixel coords
        (j1, i1) = self.wcs2pix_0CRVAL(ylim[0], xlim[0])
        (j2, i2) = self.wcs2pix_0CRVAL(ylim[1], xlim[1])

        iis = [i1, i2]
        (i1m, i2m) = map(lambda i: int(round(i)), sorted(iis))
        jjs = [j1, j2]
        (j1m, j2m) = map(lambda j: int(round(j)), sorted(jjs))

        # Clamp to image boundaries
        ny, nx = im.shape
        i1m = max(0, i1m)
        j1m = max(0, j1m)
        i2m = min(nx, i2m)
        j2m = min(ny, j2m)
        
        # Handle cases where the view is completely outside the image
        if i1m >= i2m or j1m >= j2m:
            print("Warning: Current view is outside of the image data.")
            self.selected_subim = None
            return

        subim = im[j1m:j2m, i1m:i2m]
        
        self.selected_subim = subim
        self.selected_i1m = i1m
        self.selected_j1m = j1m
        self.selected_i2m = i2m
        self.selected_j2m = j2m

        # Calculate statistics for the new region
        if subim.size > 0:
            mask = np.isfinite(subim)
            finite_vals = subim[mask]
            if finite_vals.size > 0:
                self.selected_range2 = np.max(finite_vals)
                self.selected_range1 = np.min(finite_vals)
                self.selected_rms = np.std(finite_vals)
            else: # All NaNs
                self.selected_range1, self.selected_range2, self.selected_rms = None, None, None
        else: # No data in view
             self.selected_range1, self.selected_range2, self.selected_rms = None, None, None
        
        self.subim_max = np.max(im[np.where(np.isfinite(im))])

    def on_select(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        im = self.hdu.data
        (j1, i1) = self.wcs2pix_0CRVAL(y1, x1)
        (j2, i2) = self.wcs2pix_0CRVAL(y2, x2)
        iis = [i1, i2]
        (i1m, i2m) = map(lambda i: int(i), sorted(iis))
        jjs = [j1, j2]
        (j1m, j2m) = map(lambda j: int(j), sorted(jjs))

        subim = im[j1m:j2m, i1m:i2m]

        # A single click or double-click can create a zero-size subim, which causes numpy funcs to fail.
        if subim.size == 0:
            if eclick.dblclick:
                self.rs.set_visible(False)
                self.rs.set_active(False)
                print("RectangleSelector deactivated")
            return  # Ignore zero-size selections

        diagdistance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f): Diag %3.2f" %
            (x1, y1, x2, y2, diagdistance))
        self.selected_subim = subim
        self.selected_i1m = i1m
        self.selected_j1m = j1m
        self.selected_i2m = i2m
        self.selected_j2m = j2m
        self.selected_range2 = np.max(subim[np.where(np.isfinite(subim))])
        self.selected_range1 = np.min(subim[np.where(np.isfinite(subim))])
        self.subim_max = np.max(im[np.where(np.isfinite(im))])

        mask = np.isfinite(subim)
        if np.any(np.invert(mask)):
            print("some pixels are nans")
        self.selected_rms = np.std(subim[mask])
        print("Flux: %.3e  Rms: %.3e  Min: %.3e Max: %.3e Median: %.3e PSNR: %.3e" %
            (np.sum(subim[mask]), self.selected_rms, np.min(subim[mask]), np.max(subim[mask]),
            np.median(subim[mask]), self.subim_max / self.selected_rms))
        if self.beam:
            print("Units /beam, Flux is ",np.sum(subim)/self.beam)
            
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

        self.ax.set_aspect('equal', adjustable='box')
        self.fig.canvas.draw()

    def _cleanup_fit_artists(self):
        for artist in self.fit_artists:
            artist.remove()
        self.fit_artists = []
        self.ax.set_xlim(self.xlim_store)
        self.ax.set_ylim(self.ylim_store)
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.canvas.draw_idle()

    def on_interactive_fit_motion(self, event):
        if not event.inaxes or self.fit_setup_stage not in ['fwhm_maj', 'fwhm_min']:
            return

        # Artists list contains: cross, [maj_line], [temp_lines...]
        while len(self.fit_artists) > (2 if self.fit_setup_stage == 'fwhm_min' else 1):
             self.fit_artists.pop().remove()

        cx = self.interactive_init_phys["_centroid_x_data"]
        cy = self.interactive_init_phys["_centroid_y_data"]
        x, y = event.xdata, event.ydata

        # Draw dynamic line from centroid to cursor
        line = self.ax.plot([cx, x], [cy, y], 'g-')[0]
        self.fit_artists.append(line)

        if self.fit_setup_stage == 'fwhm_min':
            pa_rad = np.radians(self.interactive_init_phys['pa_deg'])
            v_maj_unit = np.array([np.sin(pa_rad), np.cos(pa_rad)])
            v_min_unit = np.array([v_maj_unit[1], -v_maj_unit[0]])

            v_click = np.array([x - cx, y - cy])
            hwhm_min = np.abs(np.dot(v_click, v_min_unit))
            
            # Draw projected minor axis line
            min_axis_end = [cx + v_min_unit[0] * hwhm_min, cy + v_min_unit[1] * hwhm_min]
            ortho_line = self.ax.plot([cx, min_axis_end[0]], [cy, min_axis_end[1]], 'g--')[0]
            self.fit_artists.append(ortho_line)

        self.fig.canvas.draw_idle()

    def on_interactive_fit_click(self, event):
        if not event.inaxes: return

        if self.fit_setup_stage == 'centroid':
            x_cen, y_cen = event.xdata, event.ydata
            j_pix, i_pix = self.wcs2pix_0CRVAL(y_cen, x_cen)
            i_pix, j_pix = int(round(i_pix)), int(round(j_pix))
            amplitude = np.squeeze(self.hdu.data)[j_pix, i_pix]
            
            wcs_full = WCS(self.hdu.header)
            ra_cen, dec_cen = wcs_full.wcs_pix2world(i_pix, j_pix, 0)

            self.interactive_init_phys = {
                "amplitude": amplitude, "ra": float(ra_cen), "dec": float(dec_cen),
                "_centroid_x_data": x_cen, "_centroid_y_data": y_cen
            }
            
            cross = self.ax.plot(x_cen, y_cen, 'g+', markersize=10)[0]
            self.fit_artists.append(cross)
            self.ax.set_xlim(self.xlim_store)
            self.ax.set_ylim(self.ylim_store)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw_idle()

            self.fit_setup_stage = 'fwhm_maj'
            print("Centroid selected. Click to define major axis HWHM and PA.")

        elif self.fit_setup_stage == 'fwhm_maj':
            x_maj, y_maj = event.xdata, event.ydata
            cx, cy = self.interactive_init_phys["_centroid_x_data"], self.interactive_init_phys["_centroid_y_data"]
            
            # Finalize major axis line
            while len(self.fit_artists) > 1: self.fit_artists.pop().remove()
            line = self.ax.plot([cx, x_maj], [cy, y_maj], 'g-')[0]
            self.fit_artists.append(line)
            self.ax.set_xlim(self.xlim_store)
            self.ax.set_ylim(self.ylim_store)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw_idle()

            fwhm_maj = np.sqrt((x_maj - cx)**2 + (y_maj - cy)**2) * 2
            pa_deg = np.degrees(np.arctan2(x_maj - cx, y_maj - cy))
            
            self.interactive_init_phys['fwhm_maj_arcsec'] = fwhm_maj
            self.interactive_init_phys['pa_deg'] = pa_deg
            
            self.fit_setup_stage = 'fwhm_min'
            print("Major axis defined. Click to define minor axis HWHM.")

        elif self.fit_setup_stage == 'fwhm_min':
            x_min, y_min = event.xdata, event.ydata
            cx, cy = self.interactive_init_phys["_centroid_x_data"], self.interactive_init_phys["_centroid_y_data"]

            v_min_click = np.array([x_min - cx, y_min - cy])
            pa_rad = np.radians(self.interactive_init_phys['pa_deg'])
            v_maj_unit = np.array([np.sin(pa_rad), np.cos(pa_rad)])
            v_min_unit = np.array([v_maj_unit[1], -v_maj_unit[0]])

            fwhm_min = np.abs(np.dot(v_min_click, v_min_unit)) * 2
            self.interactive_init_phys['fwhm_min_arcsec'] = fwhm_min

            print("Initial parameters defined. Starting fit...")
            self.fig.canvas.mpl_disconnect(self.event_handler_id)
            self.fig.canvas.mpl_disconnect(self.motion_event_handler_id)
            self.event_handler_id = None
            self.motion_event_handler_id = None
            self.fit_setup_stage = None
            self._cleanup_fit_artists()

            del self.interactive_init_phys["_centroid_x_data"]
            del self.interactive_init_phys["_centroid_y_data"]

            self.run_fit(self.interactive_init_phys)

    def run_fit(self, init_phys):
        import mgauss
        print("Fitting with mgauss.fit")

        # Case 1: A region is selected
        if self.selected_subim is not None:
            hdr = self.hdu.header
            hduz = fits.PrimaryHDU()
            hduz.data = self.selected_subim
            hdrz = deepcopy(hdr)
            hdrz['CRPIX1'] = hdr['CRPIX1'] - self.selected_i1m
            hdrz['CRPIX2'] = hdr['CRPIX2'] - self.selected_j1m
            hduz.header = hdrz

            im_data = hduz.data
            wcs_data = WCS(hduz.header)
            
            if 'CDELT1' in hduz.header and hduz.header['CDELT1'] != 0:
                pixscale = np.abs(hduz.header["CDELT1"]) * 3600.0
            else:
                pixscale = 1.0

            rms0 = self.selected_rms
            
            model_sub, fit_params = mgauss.fit(
                im_data=im_data,
                wcs_data=wcs_data,
                pixscale=pixscale,
                rms0=rms0,
                init_phys=init_phys,
                with_plot=False
            )

            # Re-evaluate best-fit model on the full grid
            full_im_data = np.squeeze(self.hdu.data)
            full_hdr = self.hdu.header
            full_wcs = WCS(full_hdr)
            full_shape = full_im_data.shape

            if 'CDELT1' in full_hdr and full_hdr['CDELT1'] != 0:
                full_pixscale = np.abs(full_hdr["CDELT1"]) * 3600.0
            else:
                full_pixscale = 1.0

            fit_x_pix_full, fit_y_pix_full = full_wcs.wcs_world2pix(fit_params['ra'], fit_params['dec'], 0)

            L11_pix, L21_pix, L22_pix = mgauss.physical_to_cholesky(
                fit_params['fwhm_maj_arcsec'] / full_pixscale,
                fit_params['fwhm_min_arcsec'] / full_pixscale,
                fit_params['pa_deg']
            )

            full_model = mgauss.evaluate_gaussian_cholesky(
                full_shape,
                fit_params['amplitude'],
                float(fit_x_pix_full),
                float(fit_y_pix_full),
                L11_pix, L21_pix, L22_pix
            )
            
            self.last_fit_model = full_model
            self.last_fit_is_subregion = True

        # Case 2: No region is selected, fit full image
        else:
            print("No region selected, fitting full image.")
            im_data = np.squeeze(self.hdu.data)
            hdr = self.hdu.header
            wcs_data = WCS(hdr)

            if 'CDELT1' in hdr and hdr['CDELT1'] != 0:
                pixscale = np.abs(hdr["CDELT1"]) * 3600.0
            else:
                pixscale = 1.0
            
            rms0 = np.std(im_data[np.isfinite(im_data)])
            
            model, fit_params = mgauss.fit(
                im_data=im_data,
                wcs_data=wcs_data,
                pixscale=pixscale,
                rms0=rms0,
                init_phys=init_phys,
                with_plot=True
            )

            self.last_fit_model = model
            self.last_fit_is_subregion = False

        # Common post-fit logic
        wcs_full = WCS(self.hdu.header)
        x_pix, y_pix = wcs_full.wcs_world2pix(fit_params['ra'], fit_params['dec'], 0)
        d_cen_arcsec, a_cen_arcsec = self.pix2wcs_0CRVAL(y_pix, x_pix)

        ellipse = Ellipse(xy=(a_cen_arcsec, d_cen_arcsec),
                          width=fit_params['fwhm_maj_arcsec'],
                          height=fit_params['fwhm_min_arcsec'],
                          angle=90 - fit_params['pa_deg'],
                          edgecolor='green', fc='None', lw=2)
        self.last_fit_ellipse = self.ax.add_patch(ellipse)
        self.fig.canvas.draw()
        print("Fit complete. Press 'y' to subtract or 'n' to cancel.")

    def on_key_press(self, event):
        if event.key in ['C', 'c']:
            print(
                "COLOR MAPS: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r"
            )
            cmap = input('Enter new color map> ')
            self.theimage.set_cmap(cmap)
            print('Switch to ', cmap)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw()

        if event.key in ['L', 'l']:
            print("looping over all available color maps")

            acmap = cmaps[self.icmap]
            self.icmap += 1
            if self.icmap >= len(cmaps):
                self.icmap = 0
            print("trying ", acmap)
            self.theimage.set_cmap(acmap)
            print('Switch to ', acmap)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw()

        if event.key in ['A', 'a'] and not self.rs.active:
            print('RectangleSelector activated.')
            self.rs.set_active(True)
            self.rs.set_visible(True)

        if event.key in ['w', 'W']:
            if self.selected_subim is not None:
                print("Writing selection to view.fits")
                hdr = self.hdu.header
                hduz = fits.PrimaryHDU()
                hduz.data = self.selected_subim
                hdrz = deepcopy(hdr)
                hdrz['CRPIX1'] = hdr['CRPIX1'] - self.selected_i1m
                hdrz['CRPIX2'] = hdr['CRPIX2'] - self.selected_j1m
                hduz.header = hdrz
                hduz.writeto('view.fits', overwrite=True)
            else:
                print("No selection made yet. Press 'a' and select a region first.")

        if event.key in ['u', 'U']:
            if self.selected_subim is not None:
                print("Updating intensity scale to selection min/max.")
                self.theimage.set_clim(vmin=self.selected_range1, vmax=self.selected_range2)
                self.fig.canvas.draw()
            else:
                print("No region selected. Press 'a' to select a region first.")

        if event.key in ['f', 'F']:
            if self.event_handler_id is not None:
                print("Already in interactive fit setup. Press Esc to cancel.")
                return
            print("Starting interactive fit setup. Press Esc to cancel.")
            print("Click to select centroid.")
            self.xlim_store = self.ax.get_xlim()
            self.ylim_store = self.ax.get_ylim()
            if self.rs.active: self.rs.set_active(False)
            self.fit_setup_stage = 'centroid'
            self.interactive_init_phys = {}
            self.event_handler_id = self.fig.canvas.mpl_connect(
                'button_press_event', self.on_interactive_fit_click
            )
            self.motion_event_handler_id = self.fig.canvas.mpl_connect(
                'motion_notify_event', self.on_interactive_fit_motion
            )

        if event.key in ['y', 'Y', 'n', 'N'] and self.last_fit_ellipse is not None:
            if event.key in ['y', 'Y']:
                print("Subtracting model...")
                full_image = np.squeeze(self.hdu.data)
                full_image -= self.last_fit_model
                
                self.theimage.set_data(full_image)
                new_vmin, new_vmax = np.percentile(full_image[np.isfinite(full_image)], [1, 99])
                self.theimage.set_clim(vmin=new_vmin, vmax=new_vmax)
            else: # 'n' or 'N'
                print("Canceling subtraction.")

            self.last_fit_ellipse.remove()
            self.last_fit_model = None
            self.last_fit_ellipse = None
            self.fig.canvas.draw()

        if event.key == 'escape':
            if self.event_handler_id is not None:
                self.fig.canvas.mpl_disconnect(self.event_handler_id)
                self.fig.canvas.mpl_disconnect(self.motion_event_handler_id)
                self.event_handler_id = None
                self.motion_event_handler_id = None
                self.fit_setup_stage = None
                self._cleanup_fit_artists()
                self.ax.set_xlim(self.xlim_store)
                self.ax.set_ylim(self.ylim_store)
                self.ax.set_aspect('equal', adjustable='box')
                print("Interactive fit setup cancelled.")

        if event.key in ['H', 'h']:
            print('key a: activate RectangleSelector (dbleclick deactivates)')
            print('key u: update intensity scale to selection')
            print('key c: change colormap')
            print('key l: loop through colormaps')
            print('key w: save selected region to view.fits')
            print('key f: start interactive 2D Gaussian fit')
            print('key y/n: confirm/cancel Gaussian subtraction')

    def show(self):
        plt.show()
        plt.close('all')

def View(indata, cmap='RdBu_r', AllContours=False,cmapcontours='Greens_r', contlevels=[0.2, 0.4, 0.6, 0.8]):
    viewer = VtoolsViewer(indata, cmap, AllContours, cmapcontours, contlevels)
    viewer.show()

def Spec(indata, labels=False):
    if not isinstance(indata, list):  #
        indata = [
            indata,
        ]

    global fig1
    global ax1
    fig1, ax1 = plt.subplots()

    for ispec, aspec in enumerate(indata):
        if (labels):
            linelabel = labels[ispec]
        else:
            linelabel = str(ispec)
        theplot = ax1.plot(aspec[:, 0], aspec[:, 1], label=linelabel)

    plt.legend()
    plt.show()
