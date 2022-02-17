#!/usr/bin/env python

# Copyright (c) 2022
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2022


import numpy as np
import matplotlib.pyplot as plt

day = 86400                 # seconds in one day
year = 365 * day            # seconds in one year
lightspeed = 299792458      # speed of light in (m/s)
g = 9.81                    # gravity of earth  (m/s)/s = m / (s^2)
d_mars=225e+9               # average distance to Mars

t = np.linspace(0,  3 * day, 100)
v = g * t
x = v * t


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

ax1.hlines(y=lightspeed, xmin=0, xmax=t.max(),
           linestyles='--', colors='k', lw=2, label='Speed of light')
ax1.plot(t, v, 'r')
ax1.set_title('Speed')

ax2.hlines(y=d_mars, xmin=0, xmax=t.max(),
           linestyles='--', colors='k', lw=2, label='Distance to Mars')
ax2.plot(t, x, 'b')
ax2.set_title('Distance')

plt.show()
