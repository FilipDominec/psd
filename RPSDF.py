#!/usr/bin/python3  
#-*- coding: utf-8 -*-

## Import common moduli
import matplotlib, sys, os, time
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, hbar, pi
from scipy.misc import imread

## Load data
#x,y = np.loadtxt(sys.argv[1], unpack=True)

N_FREQ_BINS = 50
imname = sys.argv[1]
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

PMT_preamp_codes  = {'A':1, 'B':4, 'C':4**2, 'D':4**3, 'E':4**4, 'F':4**5}   # PMT scales roughly exponentially

im_size_code = imname[3].upper()
im_xsize, im_ysize = SEM_image_sizes[im_size_code]        # unit: meter
im_kv_code = imname[4:6]
im_pmtpreamp_code = imname[6]


im = imread(imname)
fim = np.fft.fftshift(np.fft.fft2(im))
fim2 = np.abs(fim**2)

## Generate circular domains in the 2D frequency space
xfreq = np.fft.fftshift(np.fft.fftfreq(fim2.shape[1], d=im_xsize/fim2.shape[1])) * 2*np.pi
yfreq = np.fft.fftshift(np.fft.fftfreq(fim2.shape[0], d=im_ysize/fim2.shape[0])) * 2*np.pi
mesh = np.meshgrid(xfreq,yfreq)
xyfreq = np.abs(mesh[0] + 1j*mesh[1])
max_xyfreq = np.max(xyfreq)

freq_bin_width = max_xyfreq/N_FREQ_BINS
freq_bins =  np.linspace(0, max_xyfreq, N_FREQ_BINS+1)
xyfreq_binned = np.round(xyfreq/freq_bin_width)*freq_bin_width

bin_averages = []
for freq_bin in freq_bins:
    bin_mask     = np.isclose(xyfreq_binned, freq_bin)
    bin_px_count = np.sum(bin_mask)
    bin_average  = np.sum(fim2[bin_mask])/bin_px_count * im_xsize * im_ysize  # multiply by px area, since considering energy!
    #print('binning', freq_bin, ' [m^-1] with # of px = ', bin_px_count, ' with average PSD = ', bin_average)
    bin_averages.append(bin_average)

## ==== Outputting ====
np.savetxt(sys.argv[1]+"_RPSDF.dat", np.array([freq_bins[1:], bin_averages[1:]]).T)

#plt.imshow(xyfreq_binned)
plt.plot(freq_bins[1:], bin_averages[1:])
plt.yscale('log')
plt.xscale('log')

## Finish the plot + save 
plt.xlabel(u"spatial frequency (1/m)"); 
plt.ylabel(u"spectral power (A. U.)"); 
plt.title(sys.argv[1]); 
plt.grid()
#plt.legend(prop={'size':10}, loc='upper right')
plt.savefig(sys.argv[1]+"_RPSDF.png", bbox_inches='tight')

plt.plot(freq_bins[1:], bin_averages[1:])

