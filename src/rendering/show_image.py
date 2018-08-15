import opengl as gl
from scipy import misc
import time
f = misc.face()
viewer = gl.SimpleImageViewer()
viewer.imshow(f)
while True:
    time.sleep(0.1)
