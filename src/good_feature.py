#! /usr/bin/env python3
#
# Copyright (C) 2016 Mathew Woodyard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from pylab import savefig

# Greyhounds are usually taller than Labradors.
# Eye color does not depend on the breed.
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# Visualize dog heights.
sns.distplot(grey_height, color='red', label='Greyhound')
sns.distplot(lab_height, color='blue', label='Labrador', axlabel='Dog Height')
sns.plt.legend(handles=[mpatches.Patch(color='red', label='Greyhound'),
                        mpatches.Patch(color='blue', label='Labrador')])
savefig('../out/dog_height_hist.png')

