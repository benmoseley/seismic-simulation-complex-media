from matplotlib.colors import LinearSegmentedColormap
import numpy as np

rkb = {'red':     ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

rgb = {'red':     ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

rkb = LinearSegmentedColormap('RedBlackBlue', rkb)
rgb = LinearSegmentedColormap('RedGreenBlue', rgb)


def fig2rgb_array(fig, expand=False):
    fig.canvas.draw()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(shape)