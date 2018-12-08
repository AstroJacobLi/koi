from __future__ import division
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

def manga_SBP_single(ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0, 
    x_min=1.0, x_max=40.0, alpha=1, physical_unit=False, show_dots=False, 
    vertical_line=False, vertical_pos=100, linecolor='firebrick', linestyle='-', label='SBP'):
    """Display the 1-D profiles."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.88])
        ax1.tick_params(direction='in')
    else:
        ax1 = ax
        ax1.tick_params(direction='in')

    # Calculate physical size at this redshift
    import slug
    phys_size = slug.phys_size(redshift,is_print=False)

    # 1-D profile
    if physical_unit is True:
        x = ell_fix['sma']*pixel_scale*phys_size
        y = -2.5*np.log10((ell_fix['intens'] + offset)/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens'] + offset + ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens'] + offset - ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$R/\mathrm{kpc}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma']*pixel_scale
        y = -2.5*np.log10((ell_fix['intens'] + offset)/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens'] + offset + ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens'] + offset - ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$R/\mathrm{arcsec}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    
    # ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    
    if show_dots is True:
        ax1.errorbar((x), 
                 y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)
    if label is not None:
        ax1.plot(x, y, color=linecolor, linewidth=4, linestyle=linestyle,
             label=r'$\mathrm{'+label+'}$', alpha=alpha)
    else:
        ax1.plot(x, y, color=linecolor, linewidth=4, linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x, y_upper, y_lower, color=linecolor, alpha=0.3*alpha)
    ax1.axvline(x=vertical_pos, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], 
                    color='gray', linestyle='--', linewidth=3)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()
    if label is not None:
        ax1.legend(fontsize=25, frameon=False, loc='upper right')
    
    if physical_unit is True:
        ax4 = ax1.twiny() 
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 30, 50, 100, 150, 300]
        lin_pos = [i for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
        ax4.xaxis.set_label_coords(1, 1.05)

        ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        
        
    if vertical_line is True:
        ax1.axvline(x=vertical_pos, ymin=0, ymax=1, 
                    color='gray', linestyle='--', linewidth=3)
        
    if ax is None:
        return fig
    return ax1


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


def run_mcmc_for_radius(x, y, x_err, y_err, is_pivot=True):
    import emcee
    from scipy.stats import t
    def model(theta, x):
        return theta[0]*x+theta[1]

    def lnLikelihood(theta, x, y, xerr, yerr):
        a, b = theta
        model_value = model(theta, x)
        invy = 1/(yerr**2 + xerr**2) # +(model_value**2 * np.exp(2*lnf)))
        invy = invy/(2*np.pi)
        return -0.5*np.sum(invy*(model_value-y)**2)+0.5*np.sum(np.log(invy))

    def lnPrior(theta):
        a, b = theta
        if -5<b<5:
            return np.log(t.pdf(a, 1, 1)) + 0.0
        return -np.inf

    def lnProb(theta, x, y, xerr, yerr):
        prior = lnPrior(theta)
        if ~np.isfinite(prior):
            return -np.inf
        return prior + lnLikelihood(theta, x, y, xerr, yerr)

    x_mask = ~np.isnan(x)
    y_mask = ~np.isnan(y)
    mask = np.logical_and(x_mask, y_mask)

    x = x[mask]
    y = y[mask]
    x_err = x_err[mask]
    y_err = y_err[mask]
    x_pivot = np.median(x)
    y_pivot = np.median(y)
    print('x_pivot: ', x_pivot)
    print('y_pivot: ', y_pivot)
    if is_pivot:
        x -= x_pivot
        #y -= y_pivot


    # Least Square Method
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(y_err * y_err)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, a_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    print('a_ls: ', a_ls)
    print('b_ls: ', b_ls)



    ndim, nwalkers = 2, 100
    pos = emcee.utils.sample_ball([a_ls, b_ls], [1e-3 for i in range(ndim)], size=nwalkers)
    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnProb, args=(x, y, x_err, y_err))
    step = 1000
    pos, prob, state = sampler.run_mcmc(pos, step)

    samples = sampler.chain[:,-500:,:].reshape((-1,ndim))
    samples = sampler.flatchain
    #sampler.reset()
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
    return x, y, x_err, y_err, x_pivot, y_pivot, samples


def AGN_diagnosis(fits, this_ew, y1, x1, y2, x2, suptitle, size, show_fig=False, is_print=False):
    ############### 判断星系的类型，从两个图BPT & VO87 中分别判断 ###############
    RGmask = (this_ew <= 3)  # EW[Ha]小于3的是Retired Galaxy
    nonRGmask = (this_ew > 3)
    tot_mask  = np.ones(fits['mangaid'].shape, dtype=bool)
    
    #SF1=np.logical_and(np.logical_and(y<0.61/(x-0.05)+1.30, y<0.61/(x-0.47)+1.19),SNmask)
    SF1 = np.logical_and(
        nonRGmask,
        np.logical_and(
            x1 < 0,
            np.logical_and(y1 < 0.61 / (x1 - 0.05) + 1.30,
                           y1 < 0.61 / (x1 - 0.47) + 1.19)))
    #Composite=np.logical_and(np.logical_and(y>0.61/(x-0.05)+1.30, y<0.61/(x-0.47)+1.19),SNmask)
    Composite = np.logical_and(
        nonRGmask,
        np.logical_and(y1 > 0.61 / (x1 - 0.05) + 1.30,
                       y1 < 0.61 / (x1 - 0.47) + 1.19))
    #AGN1=np.logical_xor(np.logical_xor(SF1,SNmask),Composite)

    AGN1 = np.logical_and(
        nonRGmask,
        np.logical_xor(
            np.logical_xor(
                SF1, tot_mask),
            Composite))
    if is_print:
        print("AGN1 number is %d" % sum(AGN1))
        print("SF1 number is %d" % sum(SF1))
        print("Composite number is %d" % sum(Composite))
        print("Retired number is %d" % sum(RGmask))
        print("Total number is %d" % sum(AGN1 + SF1 + Composite + RGmask))
        print('')

    SF2 = np.logical_and(
        nonRGmask, np.logical_and(y2 < 0.72 / (x2 - 0.32) + 1.30, x2 <= 0.05))
    LINER = np.logical_and.reduce([
        nonRGmask,
        np.logical_and(np.logical_xor(SF2, tot_mask), y2 <= 1.89 * x2 + 0.76),
        np.logical_or(AGN1, Composite)])
    AGN2 = np.logical_and.reduce([
        nonRGmask,
        np.logical_and(np.logical_xor(SF2, tot_mask), y2 > 1.89 * x2 + 0.76),
        np.logical_or(AGN1, Composite)])
    if is_print:
        print("SF2 number is %d" % sum(SF2))
        print("LINER number is %d" % sum(LINER))
        print("Seyfert number is %d" % sum(AGN2))
        print("Retired number is %d" % sum(RGmask))
        print("Total number is %d" % sum(AGN2 + SF2 + LINER + RGmask))
        print('')

    if show_fig:
        fig = plt.figure(figsize=(size[0], size[1]))
        ############## 画图1 ###################
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        labels = 'SF', 'Composite', 'AGN', 'Retired'
        sizes = [sum(SF1), sum(Composite), sum(AGN1), sum(RGmask)]
        explode = (0, 0, 0, 0)  #0.1表示将Hogs那一块凸显出来
        colors = 'lawngreen', 'skyblue', 'salmon', 'gray'
        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%',
            shadow=False,
            startangle=90,
            colors=colors)
        plt.title('BPT')
        
        ############## 画图2 ###################
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        labels = 'SF', 'LINER', 'Seyfert', 'Retired'
        sizes = [sum(SF2), sum(LINER), sum(AGN2), sum(RGmask)]
        explode = (0, 0, 0, 0)  #0.1表示将Hogs那一块凸显出来
        colors = 'lawngreen', 'orange', 'r', 'gray'
        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%',
            shadow=False,
            startangle=90,
            colors=colors)
        plt.title('VO87')
        #plt.savefig('Fig1.eps', dpi=400)
        plt.suptitle(suptitle)
        plt.show()
    return AGN1, SF1, Composite, SF2, AGN2, LINER, RGmask, nonRGmask

def paper_BPT_and_RG(ring_fits, rthis_ew, rthis_x, rthis_y, rthis_x2, rthis_y2, 
                     all_fits, tthis_ew,tthis_x, tthis_y, tthis_x2, tthis_y2, savefile=None):
    '''
    Plot BPT diagram for the paper.

    Parameters:
    -----------
    rthis_ew: np.array, EW(Ha) of ring sample.
    rthis_x, rthis_y, rthis_x2, rthis_y2: np.array, corresponds to NII/Ha, OIII/Hb, SII/Ha and OIII/Hb of ring sample, respectively.
    tthis_ew: np.array, EW(Ha) of total sample.
    tthis_x, tthis_y, tthis_x2, tthis_y2: np.array, corresponds to NII/Ha, OIII/Hb, SII/Ha and OIII/Hb of total sample, respectively.
    savefile: string, the path of saving your figure.

    Return:
    -------
    BPT and VO87 diagrams.
    '''
    rAGN1, rSF1, rComposite, rSF2, rAGN2, rLINER, rRGmask, rnonRGmask = koi.AGN_diagnosis(
        ring_fits, rthis_ew, rthis_y, rthis_x, rthis_y2, rthis_x2,
        'Ha ring sample \n Using 2.6kpc aperture and its EW', [9, 4], show_fig=False, is_print=False)
    tAGN1, tSF1, tComposite, tSF2, tAGN2, tLINER, tRGmask, tnonRGmask = koi.AGN_diagnosis(
        all_fits, tthis_ew, tthis_y, tthis_x, tthis_y2, tthis_x2,
        'Total sample \n using 2.6kpc aperture and its EW', [9, 4], show_fig=False, is_print=False)
    fig = plt.figure(figsize=(26, 6.5))
    plt.rcParams['font.size'] = 20.0
    ax1 = plt.subplot2grid((1, 4), (0, 0))
    rRGmask = (rthis_ew <= 3)
    rnonRGmask = (rthis_ew > 3)
    plt.scatter(
        rthis_x[rRGmask],
        rthis_y[rRGmask],
        c='gray',
        s=5,
        marker='o',
        label='RG',
        alpha=0.5)
    plt.scatter(
        rthis_x[np.logical_and(rnonRGmask, rAGN2)],
        rthis_y[np.logical_and(rnonRGmask, rAGN2)],
        c='red',
        s=15,
        marker='o',
        label='Seyfert',
        alpha=0.5)
    plt.scatter(
        rthis_x[np.logical_and(rnonRGmask, rLINER)],
        rthis_y[np.logical_and(rnonRGmask, rLINER)],
        c='orange',
        s=15,
        marker='o',
        label='LINER',
        alpha=0.5)
    plt.scatter(
        rthis_x[np.logical_and(rnonRGmask, rComposite)],
        rthis_y[np.logical_and(rnonRGmask, rComposite)],
        c='blue',
        s=15,
        marker='o',
        label='Composite',
        alpha=0.5)
    plt.scatter(
        rthis_x[np.logical_and(rnonRGmask, rSF1)],
        rthis_y[np.logical_and(rnonRGmask, rSF1)],
        c='green',
        s=15,
        marker='o',
        label='SF',
        alpha=0.5)
    x = np.linspace(-1.28, 0.04, 100)
    y1 = 0.61 / (x - 0.05) + 1.30
    plt.plot(x, y1, 'r--', alpha=0.5)
    x = np.linspace(-2.5, 0.3, 100)
    y2 = 0.61 / (x - 0.47) + 1.19
    plt.plot(x, y2, 'b--', alpha=0.5)
    x = np.linspace(-0.18, 0.9, 100)
    #y3 = 1.05*x + 0.45
    #plt.plot(x,y3,'g--',alpha=0.5)
    plt.text(-0.8, -0.1, 'SF')
    plt.text(0.2, 0.9, 'AGN')
    plt.text(-0.25, -0.85, 'Composite')
    plt.xlim(-1.3, 0.6)
    plt.ylim(-1.2, 1.3)
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    plt.text(
        0.05 * (xmax - xmin) + xmin,
        ymin + (ymax - ymin) * 0.90,
        '(a)',
        fontsize=20)
    plt.ylabel(r'$\log$([O III]/H$\beta$)')
    plt.xlabel(r'$\log$([N II]/H$\alpha$)')
    ax1.yaxis.set_ticks_position('both')
    ax1.tick_params(direction='in')
    #plt.title('RG='+ str(sum(RGmask))+' and Non-RG='+str(sum(nonRGmask)))
    leg = plt.legend(markerscale=1.2, fontsize=13, framealpha=0.5, edgecolor='k')
    for l in leg.legendHandles:
        l.set_alpha(0.8)
    plt.grid('off')
    plt.xticks([-1,-0.5,0,0.5])

    ax2 = plt.subplot2grid((1, 4), (0, 1))
    RGmask = (rthis_ew <= 3)
    nonRGmask = (rthis_ew > 3)
    plt.scatter(
        rthis_x2[np.logical_and(rnonRGmask, rAGN2)],
        rthis_y2[np.logical_and(rnonRGmask, rAGN2)],
        c='red',
        s=15,
        marker='o',
        label='Seyfert',
        alpha=0.5)
    plt.scatter(
        rthis_x2[np.logical_and(rnonRGmask, rLINER)],
        rthis_y2[np.logical_and(rnonRGmask, rLINER)],
        c='orange',
        s=15,
        marker='o',
        label='LINER',
        alpha=0.5)
    plt.scatter(
        rthis_x2[np.logical_and(rnonRGmask, rComposite)],
        rthis_y2[np.logical_and(rnonRGmask, rComposite)],
        c='blue',
        s=15,
        marker='o',
        label='Composite',
        alpha=0.5)
    plt.scatter(
        rthis_x2[np.logical_and(rnonRGmask, rSF1)],
        rthis_y2[np.logical_and(rnonRGmask, rSF1)],
        c='green',
        s=15,
        marker='o',
        label='SF',
        alpha=0.5)
    plt.scatter(
        rthis_x2[rRGmask],
        rthis_y2[rRGmask],
        c='gray',
        s=5,
        marker='o',
        label='RG',
        alpha=0.3)
    
    x = np.linspace(-0.3, 0.5, 100)
    y1 = 1.89 * x + 0.76
    plt.plot(x, y1, 'g--', alpha=0.5)
    x = np.linspace(-2.5, 0.1, 100)
    y2 = 0.72 / (x - 0.32) + 1.30
    y3 = 0.48 / (x - 0.10) + 1.30
    plt.plot(x, y2, 'b--', alpha=0.5)
    #plt.plot(x,y3,'r--',alpha=0.5)
    
    plt.text(-1, -0.8, 'SF \& Composite')
    plt.text(-0.8, 0.9, 'Seyfert')
    plt.text(0.2, 0.7, 'LINER')
    plt.xlim(-1.2, 0.6)
    plt.ylim(-1.2, 1.3)
    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    plt.text(
        0.05 * (xmax - xmin) + xmin,
        ymin + (ymax - ymin) * 0.90,
        '(b)',
        fontsize=20)
    #plt.ylabel(r'$\log$ OIII/H$\beta$')
    plt.xlabel(r'$\log$([S II]/H$\alpha$)')
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params(direction='in')
    #plt.title('RG='+ str(sum(RGmask))+' and Non-RG='+str(sum(nonRGmask)))
    #plt.legend(frameon=False, fontsize=12)
    plt.grid('off')
    plt.xticks([-1,-0.5,0,0.5])

    ax3 = plt.subplot2grid((1, 4), (0, 2))
    tRGmask = (tthis_ew <= 3)
    tnonRGmask = (tthis_ew > 3)
    plt.scatter(
        tthis_x[tRGmask],
        tthis_y[tRGmask],
        c='gray',
        s=5,
        marker='o',
        label='RG',
        alpha=0.3)
    plt.scatter(
        tthis_x[np.logical_and(tnonRGmask, tComposite)],
        tthis_y[np.logical_and(tnonRGmask, tComposite)],
        c='blue',
        s=15,
        marker='o',
        label='Composite',
        alpha=0.5)
    plt.scatter(
        tthis_x[np.logical_and(tnonRGmask, tSF1)],
        tthis_y[np.logical_and(tnonRGmask, tSF1)],
        c='green',
        s=15,
        marker='o',
        label='SF',
        alpha=0.5)
    plt.scatter(
        tthis_x[np.logical_and(tnonRGmask, tLINER)],
        tthis_y[np.logical_and(tnonRGmask, tLINER)],
        c='orange',
        s=15,
        marker='o',
        label='LINER',
        alpha=0.5)
    plt.scatter(
        tthis_x[np.logical_and(tnonRGmask, tAGN2)],
        tthis_y[np.logical_and(tnonRGmask, tAGN2)],
        c='red',
        s=15,
        marker='o',
        label='Seyfert',
        alpha=0.5)

    x = np.linspace(-1.28, 0.04, 100)
    y1 = 0.61 / (x - 0.05) + 1.30
    plt.plot(x, y1, 'r--', alpha=0.5)
    x = np.linspace(-2.5, 0.3, 100)
    y2 = 0.61 / (x - 0.47) + 1.19
    plt.plot(x, y2, 'b--', alpha=0.5)
    x = np.linspace(-0.18, 0.9, 100)
    plt.text(-0.95, -0.65, 'SF')
    plt.text(0.2, 0.9, 'AGN')
    plt.text(-0.25, -0.85, 'Composite')
    plt.xlim(-1.3, 0.6)
    plt.ylim(-1.2, 1.3)
    xmin, xmax = ax3.get_xlim()
    ymin, ymax = ax3.get_ylim()
    plt.text(
        0.05 * (xmax - xmin) + xmin,
        ymin + (ymax - ymin) * 0.90,
        '(c)',
        fontsize=20)
    plt.xlabel(r'$\log$([N II]/H$\alpha$)')
    ax3.yaxis.set_ticks_position('both')
    ax3.tick_params(direction='in')
    #plt.title('RG='+ str(sum(tRGmask))+' and Non-RG='+str(sum(tnonRGmask)))
    #plt.legend(frameon=False, fontsize=12)
    plt.grid('off')
    plt.xticks([-1,-0.5,0,0.5])

    ax4 = plt.subplot2grid((1, 4), (0, 3))
    tRGmask = (tthis_ew <= 3)
    tnonRGmask = (tthis_ew > 3)
    plt.scatter(
        tthis_x2[tRGmask],
        tthis_y2[tRGmask],
        c='gray',
        s=5,
        marker='o',
        label='RG',
        alpha=0.3)
    plt.scatter(
        tthis_x2[np.logical_and(tnonRGmask, tAGN2)],
        tthis_y2[np.logical_and(tnonRGmask, tAGN2)],
        c='red',
        s=15,
        marker='o',
        label='Seyfert',
        alpha=0.5)
    plt.scatter(
        tthis_x2[np.logical_and(tnonRGmask, tSF1)],
        tthis_y2[np.logical_and(tnonRGmask, tSF1)],
        c='green',
        s=15,
        marker='o',
        label='SF',
        alpha=0.5)
    plt.scatter(
        tthis_x2[np.logical_and(tnonRGmask, tLINER)],
        tthis_y2[np.logical_and(tnonRGmask, tLINER)],
        c='orange',
        s=15,
        marker='o',
        label='LINER',
        alpha=0.5)
    plt.scatter(
        tthis_x2[np.logical_and(tnonRGmask, tComposite)],
        tthis_y2[np.logical_and(tnonRGmask, tComposite)],
        c='blue',
        s=15,
        marker='o',
        label='Composite',
        alpha=0.5)
    x = np.linspace(-0.3, 0.5, 100)
    y1 = 1.89 * x + 0.76
    plt.plot(x, y1, 'g--', alpha=0.5)
    x = np.linspace(-2.5, 0.1, 100)
    #y2=0.72/(x-0.32)+1.30
    #y3 = 0.48 / (x - 0.10) + 1.30
    plt.plot(x, y2, 'b--', alpha=0.5)
    #plt.plot(x, y3,'r--',alpha=0.5)
    plt.text(-1, -1.05, 'SF \& Composite')
    plt.text(-0.8, 0.9, 'Seyfert')
    plt.text(0.2, 0.7, 'LINER')
    plt.xlim(-1.2, 0.6)
    plt.ylim(-1.2, 1.3)
    xmin, xmax = ax4.get_xlim()
    ymin, ymax = ax4.get_ylim()
    plt.text(
        0.05 * (xmax - xmin) + xmin,
        ymin + (ymax - ymin) * 0.90,
        '(d)',
        fontsize=20)
    plt.xlabel(r'$\log$([S II]/H$\alpha$)')
    ax4.yaxis.set_ticks_position('both')
    ax4.tick_params(direction='in')
    #ax4.yaxis.set_label_position("right")
    #plt.title('RG='+ str(sum(tRGmask))+' and Non-RG='+str(sum(tnonRGmask)))
    #plt.legend(fontsize=12,framealpha=0.5, edgecolor='k')
    plt.grid('off')
    plt.xticks([-1,-0.5,0,0.5])
    
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.set_yticklabels([])
    ax3.get_shared_y_axes().join(ax2, ax3)
    ax3.set_yticklabels([])
    ax4.get_shared_y_axes().join(ax3, ax4)
    ax4.set_yticklabels([])
    plt.subplots_adjust(wspace=0)
    if savefile is not None:
        plt.savefig(savefile, dpi=400, bbox_inches='tight')
    plt.show()

#####################################################################

def plot_sample_distribution(
        x_arr, y_arr, z_arr, method='count',
        x_bins=25, y_bins=25, z_min=None, z_max=None,
        contour=True, nticks=5, x_lim=[8.5, 12], y_lim=[-3.3, 1.5],
        n_contour=6, scatter=True, colorbar=False, gaussian=1,
        xlabel=r'$\log (M_{*}/M_{\odot})$',
        ylabel=r'$\log (\rm{SFR}/M_{\odot}\rm{yr}^{-1})$',
        title=None,
        x_title=0.6, y_title=0.1, s_alpha=0.1, s_size=10):
    
    """Density plot."""
    from astroML.stats import binned_statistic_2d
    from scipy.ndimage.filters import gaussian_filter
    ORG = plt.get_cmap('OrRd')
    ORG_2 = plt.get_cmap('YlOrRd')
    BLU = plt.get_cmap('PuBu')
    BLK = plt.get_cmap('Greys')
    PUR = plt.get_cmap('Purples')
    GRN = plt.get_cmap('Greens')
    plt.rcParams['figure.dpi'] = 100.0
    plt.rc('text', usetex=True)
    
    if x_lim is None:
        x_lim = [np.nanmin(x_arr), np.nanmax(x_arr)]
    if y_lim is None:
        y_lim = [np.nanmin(y_arr), np.nanmax(y_arr)]

    x_mask = ((x_arr >= x_lim[0]) & (x_arr <= x_lim[1]))
    y_mask = ((y_arr >= y_lim[0]) & (y_arr <= y_lim[1]))
    x_arr = x_arr[x_mask & y_mask]
    y_arr = y_arr[x_mask & y_mask]
    z_arr = z_arr[x_mask & y_mask]

    z_stats, x_edges, y_edges = binned_statistic_2d(
        x_arr, y_arr, z_arr, method, bins=(np.linspace(8, 12, x_bins), np.linspace(-3.0, 1.5, y_bins)))

    if z_min is None:
        z_min = np.nanmin(z_stats)
    if z_max is None:
        z_max = np.nanmax(z_stats)
        
    
    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.14, right=0.93,
                        bottom=0.12, top=0.99,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    #ax1.grid(linestyle='--', linewidth=2, alpha=0.5, zorder=0)

    if contour:
        CT = ax1.contour(x_edges[:-1], y_edges[:-1],
                         gaussian_filter(z_stats.T, gaussian),
                         n_contour, linewidths=1.5,
                         colors=[BLK(0.6), BLK(0.7)],
                         extend='neither')
        #ax1.clabel(CT, inline=1, fontsize=15)
    z_stats[z_stats==0] = np.nan
    HM = ax1.imshow(z_stats.T, origin='lower',
                        extent=[x_edges[0], x_edges[-1],
                                y_edges[0], y_edges[-1]],
                        vmin=z_min, vmax=z_max,
                        aspect='auto', interpolation='none',
                        cmap=BLK)
    

    if scatter:
        ax1.scatter(x_arr, y_arr, c=z_arr, cmap='Spectral', alpha=0.3, s=s_size,
                    label='__no_label__', zorder=1)
    ax1.set_xlabel(xlabel, size=25)
    ax1.set_ylabel(ylabel, size=25)
    ax1.set_yticks([-3, -2, -1, 0, 1])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    
    if colorbar:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from matplotlib.ticker import MaxNLocator
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar_ticks = MaxNLocator(nticks).tick_values(z_min, z_max)
        cbar = plt.colorbar(HM, cax=cax, ticks=cbar_ticks)
        cbar.solids.set_edgecolor("face")
    
    if title is not None:
        ax1.text(x_title, y_title, title, size=30, transform=ax1.transAxes)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.tick_params(direction='in')
    
    return fig, z_stats, x_edges, y_edges


#####################################################################
def new_moving_average(x_arr,
                        y_arr,
                        z_arr,
                        mask,
                        mass_interval_dict,
                        mass_name,
                        x_bins=3,
                        y_bins=8,
                        step=5,
                        x_lim=(9, 12),
                        y_lim=(-13, -9),
                        num_thres=2):
    from astroML.stats import binned_statistic_2d
    for num in range(0, step-1):
        forward = ((y_lim[0] - y_lim[1]) / y_bins) / (step - 1) * num
        z_stats1, x_edges, y_edges = binned_statistic_2d(
            x_arr,
            y_arr,
            z_arr,
            'count',
            bins=(np.linspace(x_lim[0], x_lim[1], x_bins + 1),
                  np.linspace(y_lim[0] + forward, y_lim[1] + forward,
                              y_bins + 1)))
        z_stats2, x_edges, y_edges = binned_statistic_2d(
            x_arr[mask],
            y_arr[mask],
            z_arr[mask],
            'count',
            bins=(np.linspace(x_lim[0], x_lim[1], x_bins + 1),
                  np.linspace(y_lim[0] + forward, y_lim[1] + forward,
                              y_bins + 1)))
        z_stats = z_stats2 / z_stats1

        number_mask = z_stats1[mass_interval_dict[mass_name]] <= num_thres
        if num == 0:
            x_stacks = (y_edges[:-1] + y_edges[1:]) / 2
            y_stacks = z_stats[mass_interval_dict[mass_name]]
            y_stacks[number_mask] = np.nan
            # Binomial Error #
            r = 0.842
            err_stacks = r * np.sqrt(z_stats[mass_interval_dict[mass_name]] * 
                          (1 - z_stats[mass_interval_dict[mass_name]]) /
                          z_stats1[mass_interval_dict[mass_name]])
            err_stacks[number_mask] = np.nan
        else:
            x_stacks = np.vstack([x_stacks, (y_edges[:-1] + y_edges[1:]) / 2])
            y = z_stats[mass_interval_dict[mass_name]]
            y[number_mask] = np.nan
            y_stacks = np.vstack([y_stacks, y])
            # Binomial Error #
            err = r * np.sqrt(z_stats[mass_interval_dict[mass_name]] * 
                      (1 - z_stats[mass_interval_dict[mass_name]]) /
                      z_stats1[mass_interval_dict[mass_name]])
            err[number_mask] = np.nan
            err_stacks = np.vstack([err_stacks, err])
            
    return x_stacks, y_stacks, err_stacks

 #####################################################################   


#####################################################################
def RG_moving_average(x_arr,
                        y_arr,
                        z_arr,
                        mask,
                        x_bins=3,
                        y_bins=8,
                        step=5,
                        x_lim=(9, 12),
                        y_lim=(-13, -9),
                        num_thres=2):
    from astroML.stats import binned_statistic_2d
    for num in range(0, step-1):
        forward = ((y_lim[0] - y_lim[1]) / y_bins) / (step - 1) * num
        z_stats1, x_edges, y_edges = binned_statistic_2d(
            x_arr,
            y_arr,
            z_arr,
            'count',
            bins=(np.linspace(x_lim[0], x_lim[1], x_bins + 1),
                  np.linspace(y_lim[0] + forward, y_lim[1] + forward,
                              y_bins + 1)))
        z_stats2, x_edges, y_edges = binned_statistic_2d(
            x_arr[mask],
            y_arr[mask],
            z_arr[mask],
            'count',
            bins=(np.linspace(x_lim[0], x_lim[1], x_bins + 1),
                  np.linspace(y_lim[0] + forward, y_lim[1] + forward,
                              y_bins + 1)))

        z_stats = z_stats2.sum(axis=0) / z_stats1.sum(axis=0)

        number_mask = z_stats1.sum(axis=0) <= num_thres
        if num == 0:
            x_stacks = (y_edges[:-1] + y_edges[1:]) / 2
            y_stacks = z_stats
            y_stacks[number_mask] = np.nan
            # Binomial Error #
            r = 0.842
            err_stacks = r * np.sqrt(z_stats * 
                          (1 - z_stats) /
                          z_stats1.sum(axis=0))
            err_stacks[number_mask] = np.nan
        else:
            x_stacks = np.vstack([x_stacks, (y_edges[:-1] + y_edges[1:]) / 2])
            y = z_stats
            y[number_mask] = np.nan
            y_stacks = np.vstack([y_stacks, y])
            # Binomial Error #
            err = r * np.sqrt(z_stats * 
                          (1 - z_stats) /
                          z_stats1.sum(axis=0))
            err[number_mask] = np.nan
            err_stacks = np.vstack([err_stacks, err])
            
    return x_stacks, y_stacks, err_stacks

#####################################################################
#####################################################################
def moving_average(line, SSFR_bin, density, step, number_limit, rxdata, rydata, txdata, tydata, rmask, tmask):
    xset = []
    np.array(xset)
    yset = []
    np.array(yset)
    errset = []
    np.array(errset)
    for j in range(0, density):
        forward = SSFR_bin / density * j
        #bins_for_SSFR = np.linspace(-13 + forward, -9 + forward, step)
        bins_for_SSFR = np.arange(-13 + forward, -9 + forward, SSFR_bin)
        bins_for_mass = np.linspace(8, 12, step)

        H_ring_bar_SSFR, bins_for_mass, bins_for_SSFR = np.histogram2d(
            rxdata[rmask], rydata[rmask], bins=(bins_for_mass, bins_for_SSFR))
        H_ring_bar_SSFR = H_ring_bar_SSFR.T

        H_base_SSFR, bins_for_mass, bins_for_SSFR = np.histogram2d(
            txdata[tmask],
            tydata[tmask],
            bins=(bins_for_mass, bins_for_SSFR))
        H_base_SSFR = H_base_SSFR.T

        H_base_SSFR[np.isnan(H_base_SSFR)] = 0

        H_bar_fraction = H_ring_bar_SSFR / H_base_SSFR
        H_bar_fraction[np.isnan(H_bar_fraction)] = 0

        SSFRshift = (bins_for_SSFR[1] - bins_for_SSFR[0]) / 2
        if sum(H_bar_fraction[:, line] != 0) == 0:
            continue

        mass_bin_name = r'$\log$(M/M$_{\odot}$): ' + str(
            bins_for_mass[line]) + r'$\sim$' + str(bins_for_mass[line + 1])

        x = bins_for_SSFR[:-1] + SSFRshift
        y = H_bar_fraction[:, line]
        num = H_base_SSFR[:, line]
        mask = (num >= number_limit)
        xset = np.append(xset, x[mask])
        yset = np.append(yset, y[mask])

        # Binomial Error #
        z = 0.842
        N = H_base_SSFR[:, line]
        err = z * np.sqrt(H_bar_fraction[:, line] *
                          (1 - H_bar_fraction[:, line]) / N)
        errset = np.append(errset, err[mask])

    return xset, yset, errset

def all_mass_moving_average(SSFR_bin, density, step, number_limit, rxdata, rydata, txdata, tydata, rmask,
                            tmask):
    xset = []
    np.array(xset)
    yset = []
    np.array(yset)
    errset = []
    np.array(errset)
    for j in range(0, density):
        forward = SSFR_bin / density * j
        #bins_for_SSFR = np.linspace(-13 + forward, -9 + forward, step)
        bins_for_SSFR = np.arange(-13 + forward, -9 + forward, SSFR_bin)
        bins_for_mass = np.linspace(8, 12, step)

        H_ring_bar_SSFR, bins_for_mass, bins_for_SSFR = np.histogram2d(
            rxdata[rmask], rydata[rmask], bins=(bins_for_mass, bins_for_SSFR))
        H_ring_bar_SSFR = H_ring_bar_SSFR.T

        H_base_SSFR, bins_for_mass, bins_for_SSFR = np.histogram2d(
            txdata[tmask],
            tydata[tmask],
            bins=(bins_for_mass, bins_for_SSFR))
        H_base_SSFR = H_base_SSFR.T

        H_base_SSFR[np.isnan(H_base_SSFR)] = 0

        SSFRshift = (bins_for_SSFR[1] - bins_for_SSFR[0]) / 2

        mass_bin_name = '$\log$(M/M$_{\odot}$): ' + str(
            bins_for_mass[line]) + '$\sim$' + str(bins_for_mass[line + 1])

        x = bins_for_SSFR[:-1] + SSFRshift
        y = H_ring_bar_SSFR.sum(axis=1) / H_base_SSFR.sum(axis=1)
        num = H_base_SSFR.sum(axis=1)
        mask = (num >= number_limit)
        xset = np.append(xset, x[mask])
        yset = np.append(yset, y[mask])
        
        # Binomial Error #
        z = 0.842
        N = H_base_SSFR.sum(axis=1)
        err = z * np.sqrt(y *(1 - y) / N)
        errset = np.append(errset, err[mask])

    return xset, yset, errset

def normalization(newx, newy):
    xvals = np.linspace(-12.5, -9.5, 10000)
    tck = itp.splrep(newx, newy)
    def f(x):
        return itp.splev(x, tck)
    integrate = sci.integrate.quad(f, -12, -9.5)[0]
    def g(x):
        return itp.splev(x, tck) / integrate
    y_bspline = itp.splev(xvals, tck) / integrate

    print(sci.integrate.quad(g, -12, -9.5)[0])

def total_sample_moving_average(SSFR_bin, density, step, number_limit, txdata, tydata, tmask):
    xset = []
    np.array(xset)
    yset = []
    np.array(yset)
    errset = []
    np.array(errset)
    for j in range(0, density):
        forward = SSFR_bin / density * j
        #bins_for_SSFR = np.linspace(-13 + forward, -9 + forward, step)
        bins_for_SSFR = np.arange(-13 + forward, -9 + forward, SSFR_bin)
        bins_for_mass = np.linspace(8, 12, step)

        H_base_SSFR, bins_for_mass, bins_for_SSFR = np.histogram2d(
            txdata[tmask],
            tydata[tmask],
            bins=(bins_for_mass, bins_for_SSFR))
        H_base_SSFR = H_base_SSFR.T

        H_base_SSFR[np.isnan(H_base_SSFR)] = 0

        SSFRshift = (bins_for_SSFR[1] - bins_for_SSFR[0]) / 2

        mass_bin_name = '$\log$(M/M$_{\odot}$): ' + str(
            bins_for_mass[line]) + '$\sim$' + str(bins_for_mass[line + 1])

        x = bins_for_SSFR[:-1] + SSFRshift
        y = H_base_SSFR.sum(axis=1)
        num = H_base_SSFR.sum(axis=1)
        mask = (num >= number_limit)
        xset = np.append(xset, x[mask])
        yset = np.append(yset, y[mask])
        
        # Binomial Error #
        z = 0.842
        N = H_base_SSFR.sum(axis=1)
        err = z * np.sqrt(y *(1 - y) / N)
        errset = np.append(errset, err[mask])

    return xset, yset, errset
    return integrate, xvals, y_bspline