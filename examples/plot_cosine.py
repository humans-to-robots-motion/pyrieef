
import demos_common_imports
import matplotlib.pyplot as plt
import numpy

x = numpy.linspace(0., 6.28, 100)
# series = numpy.tan(x)
series = numpy.cos(x)
plt.figure()
plt.plot(series)
plt.show()
