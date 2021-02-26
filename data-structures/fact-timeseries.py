import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('outputfile')
args = parser.parse_args()

pixel = 102

f = fits.open('data-structures/drscalib.fits.gz')

raw_data = f[1].data['Data'][0]
calibrated = f[1].data['DataCalibrated'][0]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

pixel_raw = raw_data[pixel * 300 + 10:(pixel + 1) * 300]
pixel_calib = calibrated[pixel * 300 + 10:(pixel + 1) * 300]

l1, = ax1.plot(np.arange(0, 290) / 2, pixel_raw)

l2, = ax2.plot(
    np.arange(0, 290) / 2, pixel_calib,
    color='C1',
)


window_start = 40
window_width = 30
index = np.arange(window_start, window_width + window_start)
ax2.fill_between(
    index / 2,
    np.zeros_like(index),
    pixel_calib[index],
    color='C2',
    alpha=0.1, lw=0,
)
ax2.axvline(window_start / 2, color='C2')

ax1.legend((l1, l2), ['Raw Data', 'After Calibration'])

ax1.set_xlabel(r'$t \mathbin{/} \si{\nano\second}$')
ax1.set_ylabel(r'$\mathrm{ADC\,Counts}', color='C0')
ax2.set_ylabel(r'$U \mathbin{/} \si{\milli\volt}$', color='C1')
ax1.set_title(r'Run 20130102\_060, EventID 11, Pixel 104')

fig.tight_layout(pad=0)
fig.savefig(args.outputfile)
