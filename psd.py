#!/usr/bin/python3  
#-*- coding: utf-8 -*-

## Import common moduli
import matplotlib, sys, os, time, pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, hbar, pi
from imageio import imread


## TODO use settings:
N_FREQ_BINS = 500
NORMALIZE_TO_AVERAGE = True #False             # usually we care about the inhomogeneities as compared to avg. brightness
CONVERT_SPFREQ_TO_UM = False             # readers may find it more understandable to invert x-axis into metres
NOISE_BACKGROUND_CUTOFF = 2.0           # the higher, the more points will be cut
SAVE_PLOT            = 0                # diagnostic PNG


## Load data
#x,y = np.loadtxt(sys.argv[1], unpack=True)
def loadfile_SEM_XL30(imname):
    SEM_image_sizes  = {                     # magnifications
        'E':    [11740.0e-6, 8627.0e-6],              # 10       ×
        'F':    [ 5870.0e-6, 4313.5e-6],              # 20       ×
        'G':    [ 2348.0e-6, 1725.4e-6],              # 50       ×
        'H':    [ 1174.0e-6,  862.7e-6],              # 100      ×
        'I':    [  587.0e-6, 431.35e-6],              # 200      ×
        'J':    [  234.8e-6, 172.54e-6],              # 500      ×
        'K':    [  117.4e-6,  86.27e-6],              # 1000     ×
        'L':    [   58.7e-6, 43.135e-6],              # 2000     ×
        'M':    [  23.48e-6, 17.254e-6],              # 5000     ×
        'N':    [  11.74e-6,  8.627e-6],              # 10000    ×
        'O':    [   5.87e-6, 4.3135e-6],              # 20000    ×
        'P':    [  2.348e-6, 1.7254e-6],              # 50000    ×
        }

    # PMT_preamp_codes  = {'A':1, 'B':4, 'C':4**2, 'D':4**3, 'E':4**4, 'F':4**5}   # PMT scales roughly exponentially
    #im_kv_code = imname[4:6] # unused
    #im_pmtpreamp_code = imname[6] # unused

    
    try:
        im_size_code = pathlib.Path(imname).stem.upper()[3]
        im_xsize, im_ysize = SEM_image_sizes[im_size_code]        # unit: meter
    except KeyError: # perhaps different file naming convention
        im_size_code = pathlib.Path(imname).stem.upper()[2]
        im_xsize, im_ysize = SEM_image_sizes[im_size_code]        # unit: meter

    im = imread(imname)
    return im, im_xsize, im_ysize

def loadfile_WLI(imname):
    WLI_image_sizes  = {
            '5x': [1400e-6, 1050e-6], 
            '20x': [350e-6, 262e-6], 
            '50x': [140e-6, 105e-6], 
            }
    for k,v in WLI_image_sizes.items():
        if k in imname:
            im_xsize, im_ysize = WLI_image_sizes[k]        
            print('detected size', im_xsize, im_ysize)
            break
    # TODO 


all_results_freq, all_results_psd = [], []
for imname in sys.argv[1:]:
    if 'mapa' in imname:
        print(f'Processing {imname} as Horiba 2D spectral map for photoluminescence')
        raise NotImplementedError ## TODO
    elif 'mapa-raman' in imname:
        print(f'Processing {imname} as Horiba 2D spectral map for Raman')
        raise NotImplementedError ## TODO
    elif 'WLI' in imname:
        print(f'Processing {imname} as White-Light Interferometry profiler map')
        raise NotImplementedError ## TODO
    else:
        print(f'Processing {imname} as SEM XL30 image')
        im, im_xsize, im_ysize = loadfile_SEM_XL30(imname)

    print('successfully loaded '+imname+', guessing its real dimensions as ',im_xsize*1e6, '×', im_ysize*1e6, 'μm')

    fim = np.fft.fftshift(np.fft.fft2(im))
    fim2 = np.abs(fim**2)

    ## Generate circular domains in the 2D frequency space
    xfreq = np.fft.fftshift(np.fft.fftfreq(fim2.shape[1], d=im_xsize/fim2.shape[1])) * 2*np.pi
    yfreq = np.fft.fftshift(np.fft.fftfreq(fim2.shape[0], d=im_ysize/fim2.shape[0])) * 2*np.pi
    mesh = np.meshgrid(xfreq,yfreq)
    xyfreq = np.abs(mesh[0] + 1j*mesh[1])
    max_xyfreq = np.max(xyfreq/2)

    freq_bin_width = max_xyfreq/N_FREQ_BINS
    freq_bins =  np.linspace(0, max_xyfreq, N_FREQ_BINS+1)
    xyfreq_binned = np.round((xyfreq/freq_bin_width))*freq_bin_width

    bin_averages = []
    for freq_bin in freq_bins:
        bin_mask     = np.isclose(xyfreq_binned, freq_bin)
        bin_px_count = np.sum(bin_mask)
        bin_average  = np.sum(fim2[bin_mask])/bin_px_count * im_xsize * im_ysize  # multiply by px area, since considering energy!
        #print('binning', freq_bin, ' [m^-1] with # of px = ', bin_px_count, ' with average PSD = ', bin_average)
        if bin_px_count:
            bin_averages.append(bin_average)
        else:
            freq_bins.remove(freq_bin)

    ## == postprocessing ==
    if NORMALIZE_TO_AVERAGE: 
        bin_averages = bin_averages / np.mean(im)**2
        normlabel = 'normalized '
    else:
        normlabel = ''
    print('np.mean(im) = ', np.mean(im))
    if CONVERT_SPFREQ_TO_UM:
        xlabel, ylabel = u'feature size (μm)',  normlabel + u'spectral power (A.U.)'
        bin_averages *= freq_bins**2
        freq_bins = 1e6 * 2*np.pi / freq_bins
    else:
        xlabel, ylabel = u'spatial frequency $k$ (1/m)', normlabel + u'spectral power (A.U.)'

    # remove zero frequency, and all high frequencies where only noise can be expected
    bin_filter = (bin_averages > np.min(bin_averages)* NOISE_BACKGROUND_CUTOFF) 
    print(bin_averages)
    print(np.min(bin_averages))
    print(bin_filter)
    freq_bins = np.array(freq_bins)[bin_filter][1:]
    bin_averages = np.array(bin_averages)[bin_filter][1:]

    ## ==== Outputting ====
    # TODO with open(imname+'_RPSDF.dat', 'w')   as of: of.write('# ' + '\t'.join([xlabel,ylabel]) + '\n')
    # TODO with open(imname+'_RPSDF.dat', 'a+b') as of: np.savetxt(of, np.array([freq_bins, bin_averages]).T, delimiter='\t')
    np.savetxt(imname+"_RPSDF_notnorm.dat", np.array([freq_bins, bin_averages]).T)
    all_results_freq.append(freq_bins)
    all_results_psd.append(bin_averages)


# finally plot them together

for f,psd in zip(all_results_freq, all_results_psd):
    plt.plot(f, psd)
plt.yscale('log')
plt.ylim(1e-6, 1e2)
plt.xscale('log')

## Finish the plot + save 
plt.xlabel(u"spatial frequency (1/m)");  # TODO char. length = 2pi/k
plt.ylabel(u"spectral power (A. U.)");  # TODO note if normalized
plt.title(sys.argv[1]); 
plt.grid()
#plt.legend(prop={'size':10}, loc='upper right')
plt.savefig(sys.argv[1]+"_RPSDF.png", bbox_inches='tight') # TODO wise choice of output dir

plt.plot(freq_bins[1:], bin_averages[1:])
