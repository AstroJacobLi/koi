import numpy as np
import polarTransform
import peakutils.peak
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
import koi

plt.rc('font', size=20)
plt.rc('text', usetex=True)

# Calculate physical size of a given redshift
def phys_size(redshift, is_print=True, h=0.7, Omegam=0.3, Omegal=0.7):
    '''Calculate the corresponding physical size per arcsec of a given redshift.
    Requirement:
    -----------
    cosmology: https://github.com/esheldon/cosmology
    
    Parameters:
    -----------
    redshift: float
    
    Returns:
    -----------
    physical_size: float, in 'kpc/arcsec'
    '''
    import cosmology
    cosmos = cosmology.Cosmo(H0=100*h, omega_m=Omegam, flat=True, omega_l=Omegal, omega_k=None)
    ang_distance = cosmos.Da(0.0, redshift)
    physical_size = ang_distance/206265*1000 # kpc/arcsec
    if is_print:
        print ('At redshift', redshift, ', 1 arcsec =', physical_size, 'kpc')
    return physical_size


# Generate url of MaNGA Pipe3D datacube
def gen_url_manga(plate, mangaid):
    '''Generate url of MaNGA Pipe3D datacube.

    Parameters:
    -----------
    plate: int, such as 8077
    mangaid: string, such as 'manga-8077-12705'

    Return:
    -------
    url: string
    '''
    return [
        'https://data.sdss.org/sas/dr14/manga/spectro/pipe3d/v2_1_2/2.1.2/'
        + str(plate)
        + '/' + mangaid + '.Pipe3D.cube.fits.gz'
    ]

# Calculate the mean profile and its errors in a fan-shaped area (which I called 'pizza')

def eat_pizza(init, theta, polarimage, xinput, r_max, dx, phys_scale, pa, ba):
    '''Calculate the mean profile and its errors in a fan-shaped area (which I called 'pizza').

    Parameters:
    -----------
    init: float, angle of starting point of the pizza
    theta: float, angle ot the fan-shaped area (pizza)
    polarimage: 2-D np.array, the polar-transformed image, usually returned by `polarTransform.convertToPolarImage`
    xinput: 1-D np.array, usually ranges from 0 to r_max * phys_scale. The unit is `kpc`.

    Return:
    -------
    : mean profile points corresponding to the `xinput`
    : std value of the mean profile
    '''
    from scipy import interpolate
    for i in range(init, init + theta):
        phi = np.deg2rad(i)
        x = (np.arange(0, r_max) * dx * phys_scale * (np.sqrt((np.cos(pa - phi))**2 + ((np.sin(pa - phi) / ba)**2))))
        y = polarimage[:, i%360]
        f = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
        if i==init:
            ystack = f(xinput)
        else:
            ystack = np.vstack((ystack, f(xinput)))
    return ystack.mean(axis=0), np.std(ystack, axis=0)

    


# Find nearset position
def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx


def show_ha_profile(obj, x_input, y_output, y_std, r_max):
    fig, [ax1, ax2] = plt.subplots(1, 2 ,figsize=(16, 6))
    plt.rc('font', size=20)
    plt.rc('text', usetex=True)
    cubeset = fits.open('/Users/jiaxuanli/Research/MaNGA/v2_1_2/' + obj['mangaid'].rstrip(' ') + '.Pipe3D.cube.fits.gz')
    ha_im = cubeset[3].data[45]            # H-alpha image
    mask = np.isnan(cubeset[1].data[18])
    dx = -cubeset[3].header['CD1_1']*3600
    dy = cubeset[3].header['CD2_2']*3600
    x_center = np.int(cubeset[3].header['CRPIX1']) - 1
    y_center = np.int(cubeset[3].header['CRPIX2']) - 1
    x_extent = (np.array([0., ha_im.shape[0]]) - (ha_im.shape[0] - x_center)) * dx * (-1)
    y_extent = (np.array([0., ha_im.shape[1]]) - (ha_im.shape[1] - y_center)) * dy
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]] # arcsec
    phys_scale = koi.phys_size(obj['redshift'], h=0.71, Omegam=0.27, Omegal=0.73)
    cubeset.close()

    ## ax1
    im1 = ax1.imshow(np.ma.array(ha_im, mask=mask), origin='lower', extent=extent, cmap='Spectral_r')
    fig.colorbar(im1, label=r'$10^{-16}\ \mathrm{erg/s/cm}^2$', fraction=0.045, pad=0.04, ax=ax1)

    tks = [5 * i for i in range(-int((round(extent[0]/5) - 1)), int(round(extent[0]/5)))]
    ax1.set_yticks(tks)
    ax1.set_xticks(tks)
    ax1.text(0.5, 0.03, r'$\mathrm{H\alpha\ flux},\ \texttt{' + obj['mangaid'] + '}$', 
            bbox=dict(edgecolor='k', alpha=0.1),
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax1.transAxes)
    ax1.set_xlabel(r'$\mathrm{arcsec}$')
    ax1.set_ylabel(r'$\mathrm{arcsec}$')
    ax1.tick_params(direction='in')
    ax1.set_ylim(1.2*extent[1], -0.98*extent[1])
    #plt.title(r'H$\alpha$ Flux of '+id)
    #angscale = 360 / polarImage.shape[1] # degree/pix, for the polarImage
    angscale = 1

    ## ax2
    y_upper = y_output + y_std
    y_lower = y_output - y_std
    profile, = ax2.plot(x_input, y_output, linewidth=2, linestyle='-', c="firebrick", zorder=9)
    ax2.fill_between(x_input, y_upper, y_lower, color='orangered', alpha=0.3)

    peak_indices = peakutils.peak.indexes(y_output, thres=0, min_dist=0)
    #scatter_high = ax2.scatter(x_input[peak_indices[:2]], y_output[peak_indices[:2]], zorder=10,
    #                s=200, marker=(5,1,0), facecolors='yellow', edgecolors='red')
    if len(peak_indices)==0:
        peaks_reliable = np.nan
    else:
        peaks = x_input[peak_indices]
        
        reliable_mask = (abs(peaks - r_max * dx * phys_scale / 2) < r_max * dx * phys_scale / 3)
        if sum(reliable_mask) > 0:
            reliable_inx = np.argmax(y_output[peak_indices[reliable_mask]])
            
            peaks_reliable = peaks[reliable_mask][reliable_inx]
            #reliable_inx = np.where(x_input == peaks_reliable)[0][0]
            scatter_high = ax2.scatter(peaks_reliable, np.max(y_output[peak_indices[reliable_mask]]), zorder=10,
                            s=200, marker=(5,1,0), facecolors='yellow', edgecolors='red')
            print('peak reliable', peaks_reliable)
        else: peaks_reliable = np.nan
        

    trough_indices = peakutils.peak.indexes(1 - y_output, thres=0, min_dist=0)
    scatter_low = ax2.scatter(x_input[trough_indices[0]], y_output[trough_indices[0]], zorder=10,
                    s=200, marker=(5,1,0), facecolors='lawngreen', edgecolors='blue')
    #trough_set = np.dstack((x_input[trough_indices], y_output[trough_indices]))
    troughs = x_input[trough_indices]
    troughs_reliable = troughs[0]
    print('trough reliable', troughs[0])

    # PSF position
    PSFposition = phys_scale * cubeset[3].header['RFWHM']
    PSF = ax2.fill_between(np.array(x_input)[x_input < PSFposition], 
                           y_lower[x_input < PSFposition], 
                           0, facecolor='gray', alpha=0.15, interpolate=True, zorder=0)

    # Legend
    lines = ax2.get_lines()
    #legend1= plt.legend([lines[i] for i in [0,1]], [r'Deprojected H$\alpha$ profile',r'Smoothed H$\alpha$ profile'], 
    #                    loc='upper left', frameon=False)
    legend1= plt.legend([profile], [r'$\mathrm{H\alpha\ flux\ profile}$'], 
                        loc='upper left', frameon=False, bbox_to_anchor=(0.1,0.99))
    if sum(reliable_mask)!=0:
        legend2= plt.legend([scatter_high, scatter_low, PSF], 
                            map(lambda t: r'$\mathrm{' + t + '}$', ['Peak','Trough','PSF']), 
                            loc='upper right', 
                            frameon=False) # bbox_to_anchor=(0.9,0.75)
        ax2.add_artist(legend2)

    ax2.add_artist(legend1)
    ax2.tick_params(direction='in')
    plt.xlabel(r'$\mathrm{Radial\ distance\ (kpc)}$')
    plt.ylabel(r'$\mathrm{H\alpha\ flux\ (}$'+r'$10^{-16}\ \mathrm{erg/s/cm}^2)$', fontsize=20)
    plt.legend(frameon=False)

    plt.ylim(0, 1.15*max(y_output))
    plt.xlim(0, r_max * phys_scale * dx)
    plt.subplots_adjust(wspace=0.4)
    return fig, [ax1, ax2], peaks_reliable, troughs_reliable


