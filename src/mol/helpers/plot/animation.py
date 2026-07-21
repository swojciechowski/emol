import matplotlib.pyplot as plt
import numpy as np
import imageio
import imageio.v3 as iio
import os
import tempfile


class Animation:
    def __init__(self, tmp_path=None):
        self.tmp_path = tmp_path

        if self.tmp_path is None:
            self.tmpdir = tempfile.TemporaryDirectory()
            self.tmp_path = self.tmpdir.name

        self._frame_counter = 0

    def __del__(self):
        if getattr(self, 'tmpdir', None):
            self.tmpdir.cleanup()

    def add_frame(self, figure=None):
        if figure is not None:
            figure_tmp = plt.gcf().number
            plt.figure(figure.number)

        plt.savefig(os.path.join(self.tmp_path, f"{self._frame_counter}.png"))

        if figure is not None:
            plt.figure(figure_tmp)

        self._frame_counter += 1

    def export(self, out_name, delay=10):
        frames = np.stack([iio.imread(os.path.join(self.tmp_path, f"{x}.png")) for x in range(self._frame_counter)], axis=0)
        imageio.mimsave(out_name, frames, format='GIF', loop=0, duration=delay)
