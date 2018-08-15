import opengl as gl
from skimage import data
from skimage.transform import resize, downscale_local_mean,  rescale
from skimage import img_as_ubyte
import warnings
import time
import matplotlib.pyplot as plt

use_matplotlib = False

image = data.chelsea()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # WARNING rescale 
    image_scaled = img_as_ubyte(rescale(image, 1.0 / 4.0))

if not use_matplotlib:
    viewer = gl.SimpleImageViewer()
    while True:
        viewer.imshow(image_scaled)
        time.sleep(0.01)
else:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[1].imshow(image_scaled)
    ax[1].set_title("Scaled image")
    plt.tight_layout()
    plt.show()
