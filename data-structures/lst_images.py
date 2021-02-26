from copy import copy
import numpy as np
from ctapipe.io import read_table
from ctapipe.instrument import SubarrayDescription
from ctapipe.visualization import CameraDisplay
import matplotlib.pyplot as plt


path = 'data-structures/lst_events.dl1.hdf5'
subarray = SubarrayDescription.from_hdf(path)

event_index = 1

images1 = read_table(path, '/dl1/event/telescope/images/tel_003')
images2 = read_table(path, '/dl1/event/telescope/images/tel_002')

fig, axs = plt.subplots(2, 2, constrained_layout=True)


cmap_image = copy(plt.get_cmap('inferno'))
cmap_image.set_bad('k')
cmap_time = copy(plt.get_cmap('RdBu_r'))
cmap_time.set_bad('lightgray')

cam = subarray.tel[1].camera.geometry

displays = [
    CameraDisplay(cam, show_frame=False, ax=axs[0, 0], cmap=cmap_image),
    CameraDisplay(cam, show_frame=False, ax=axs[0, 1], cmap=cmap_time),
    CameraDisplay(cam, show_frame=False, ax=axs[1, 0], cmap=cmap_image),
    CameraDisplay(cam, show_frame=False, ax=axs[1, 1], cmap=cmap_time),
]



displays[0].image = images1['image'][event_index]
peak_time1 = images1['peak_time'][event_index]
peak_time1[~images1['image_mask'][event_index]] = np.nan
displays[1].image = peak_time1

displays[2].image = images2['image'][event_index]
peak_time2 = images2['peak_time'][event_index]
peak_time2[~images2['image_mask'][event_index]] = np.nan
displays[3].image = peak_time2


displays[0].norm = 'log'
displays[2].norm = 'log'

for ax in axs.flat:
    ax.set_axis_off()
    ax.set_title('')

for display in displays:
    display.add_colorbar()


# displays[1].set_limits_minmax(10, 20)
axs[0, 0].set_title('Number of Photons')
axs[0, 1].set_title('Arrival Time in ns')

fig.savefig('build/data-structures/lst_images.pdf')
