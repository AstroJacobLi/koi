from __future__ import division
import numpy as np

from .__init__ import phys_size
from astropy.io import fits
from astropy.table import Table, Column

import matplotlib.pyplot as plt

def LIER_diagnosis(ew, y2, x2):
    '''
    Only use S[II] BPT diagram.
    Input:
        ew: equvalent width of H-alpha.
        y2: np.log10(OIII/Hbeta)
        x2: np.log10(SII/Halpah)
    '''
    status = 0
    
    if ew < 1:
        status = 1
    elif np.isnan(x2) or np.isnan(y2):
        status = 0
    else:
        SF2 = np.logical_and(y2 < 0.72 / (x2 - 0.32) + 1.30, x2 <= 0.05)
        LINER = np.logical_and(~SF2, y2 <= 1.89 * x2 + 0.76)
        AGN2 = np.logical_and(~SF2, y2 > 1.89 * x2 + 0.76)
        status = np.where(np.array([SF2, LINER, AGN2]))[0][0] + 2
    return status

def plot_BPT_for_each(obj, ax=None, show_fig=False, save_fig=True):
    '''
    Plot BPT diagram for each IFU data.
    Input: object, a row in catalog.
    '''
    print(obj['mangaid'])
    cubeset = fits.open('/Users/jiaxuanli/Research/MaNGA/v2_1_2/' + obj['mangaid'].rstrip(' ') + '.Pipe3D.cube.fits.gz')
    ha_im = cubeset[3].data[45]
    
    phys_scale = phys_size(obj['redshift'], is_print=False)
    dx = -cubeset[3].header['CD1_1']*3600
    dy = cubeset[3].header['CD2_2']*3600
    #print('spaxel_scale: ', dx, dy)
    x_center = np.int(cubeset[3].header['CRPIX1']) - 1
    y_center = np.int(cubeset[3].header['CRPIX2']) - 1
    x_extent = (np.array([0., ha_im.shape[0]]) - (ha_im.shape[0] - x_center)) * dx * (-1)
    y_extent = (np.array([0., ha_im.shape[1]]) - (ha_im.shape[1] - y_center)) * dy
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]] # arcsec
    #print('extent (arcsec): ', extent)

    # Fluxes
    mass_im = cubeset[1].data[18]
    ha_im = cubeset[3].data[45]
    n2_im = cubeset[3].data[46]
    hb_im = cubeset[3].data[28]
    o3_im = cubeset[3].data[26]
    ew_ha_im = - cubeset[3].data[216] * 0.5 
    # Pipe3D has an issue that EW are 2 times larger.
    s2_im = (cubeset[3].data[49] + cubeset[3].data[50])
    cubeset.close()

    # Diagnosis
    status_arr = np.zeros_like(mass_im)
    for i in range(mass_im.shape[0]):
        for j in range(mass_im.shape[1]):
            if np.isnan(mass_im[i][j]):
                status_arr[i][j] = 0
            else:
                status_arr[i][j] = LIER_diagnosis(ew_ha_im[i][j], 
                                                      np.log10(o3_im / hb_im)[i][j],
                                                      np.log10(s2_im / ha_im)[i][j])
    status_arr[status_arr == 0] = np.nan

    # Plot
    from matplotlib.colors import ListedColormap
    if ax is not None:
        [ax1, ax2] = ax
    else:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.5, 4))
    # ax1: Ha flux
    im1 = ax1.imshow(np.ma.array(ha_im, mask=np.isnan(mass_im)), 
                     origin='lower', extent=extent, cmap='Spectral_r')
    cb = plt.colorbar(im1, label=r'$10^{-16}\ \mathrm{erg/s/cm}^2$', orientation='horizontal',
                 fraction=0.037, pad=0.1, ax=ax1)
    cb.ax.tick_params(labelsize=12)

    if extent[0] >= 15:
        tks = [10 * i for i in range(-int((round(extent[0]/10) - 1)), int(round(extent[0]/10)))]
    else:
        tks = [5 * i for i in range(-int((round(extent[0]/5) - 1)), int(round(extent[0]/5)))]
    ax1.set_yticks(tks)
    ax1.set_xticks(tks)
    #ax1.set_xlabel(r'$\mathrm{arcsec}$', size=15)
    ax1.set_ylabel(r'$\mathrm{arcsec}$', size=15, labelpad=-10)
    ax1.tick_params(direction='in', labelsize=13)

    ax1.set_ylim(0.98*extent[1], -1.2*extent[1])
    ax1.text(0.5, 0.9, r'$\mathrm{H\alpha\ flux},\ \texttt{' + obj['mangaid'] + '}$', 
                bbox=dict(edgecolor='k', alpha=0.1),
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax1.transAxes, fontsize=10)
    # ax2: BPT
    cm = ListedColormap(['silver', '#abdda4', '#fdae61', '#d7191c'], name='from_list', N=None)
    cax = ax2.imshow(status_arr, origin='low', 
                     cmap=cm, vmin=0.5, vmax=4.5,
                     extent=extent)
    cbar = plt.colorbar(cax, ticks=[1, 2, 3, 4], orientation='horizontal',
                pad=0.1, fraction=0.037)
    cbar.ax.set_xticklabels(['LineLess', 'SF', 'LIER',' Seyfert'], size=13)  # horizontal colorbar
    #ax2.set_xlabel(r'$\mathrm{arcsec}$', size=15)
    #ax2.set_ylabel(r'$\mathrm{arcsec}$', size=15, labelpad=-10)
    ax2.tick_params(direction='in', labelsize=13)
    ax2.set_yticks(tks)
    ax2.set_xticks(tks)

    ax2.set_ylim(0.98*extent[1], -1.2*extent[1])
    ax2.text(0.5, 0.9, r'$\mathrm{BPT\ diagram},\ \texttt{' + obj['mangaid'] + '}$', 
                bbox=dict(edgecolor='k', alpha=0.1),
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax2.transAxes, fontsize=10)

    # Plot PSF
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy = (-extent[0]*0.8, -extent[0]*0.8), width=2.5 / dx, height=2.5 / dy, 
                  hatch='///', facecolor='none', edgecolor='k')
    ax2.add_artist(ell)

    ell = Ellipse(xy = (0, 0), width=2* 0.75 * obj['TroughRadius'] / phys_scale / dx , 
                    height=2* 0.75 * obj['TroughRadius'] / phys_scale / dx, 
                    facecolor='none', edgecolor='r')
    ax2.add_artist(ell)

    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    ax1.yaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

    plt.subplots_adjust(wspace=0)
    if save_fig:
        plt.savefig('../Figures/BPT/' + obj['mangaid'].rstrip(' ') + '.pdf', 
            dpi=200, bbox_inches='tight')
        plt.close()
    if show_fig:
        plt.show()
    if ax is not None:
        return ax
    else:
        return [ax1, ax2]