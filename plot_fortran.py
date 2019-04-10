from neuralbec import data
import numpy as np
import os

PATH = 'data/'
FILENAME = 'fort.3'


if __name__ == '__main__':

  lines = [ line.replace('\n', '').strip()
      for line in open(os.path.join(PATH, FILENAME)).readlines() ]
  x = np.array([ float(line.split()[0]) for line in lines ])
  psi = np.array([ float(line.split()[-1]) for line in lines ])
  data.plot_wave_function(x, psi, title=FILENAME,
      save_to='{}.png'.format(FILENAME)
      )
