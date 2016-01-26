from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt


class ScaleSpace(list):
    def __init__(self, orig_image, sigma=1.6, s=3, extra=3, min_width=32, min_height=32, *args):
        super().__init__(*args)
        self.sigma = 1.6
        self.s = 3
        self.k = self._calc_k(self.s)
        self.extra = extra
        self.orig_image = orig_image
        self.min_width = min_width
        self.min_height = min_height

    def _calc_k(self, s):
        return np.power(2, 1/self.s)

    def create(self, orig_image=None, sigma=None, s=None, extra=None, min_width=None, min_height=None):
        orig_image = orig_image if orig_image is not None else self.orig_image
        sigma = sigma if sigma else self.sigma
        s = s if s else self.s
        k = self._calc_k(s) if s else self.k
        extra = extra if extra else self.extra
        min_width = min_width if min_width else self.min_width
        min_height = min_height if min_height else self.min_height

        octave = [orig_image, ] if len(self) else []
        for n in range(len(octave), s + extra):
            octave.append(gaussian_filter(orig_image, np.power(k, n) * sigma / np.power(2, len(self))))
        self.append(octave)

        if orig_image.shape[0] / 2 >= min_height and orig_image.shape[1] / 2 >= min_width:
            self.create(np.array(octave[s][::2, ::2]))


if __name__ == '__main__':
    lena_img = Image.open('img/lena.jpg').convert('L')
    lena_img = np.array(lena_img, dtype=np.float) / 255

    scale_space = ScaleSpace(lena_img)
    scale_space.create()

    f, ax = plt.subplots(len(scale_space), len(scale_space[0]))
    for i in range(len(scale_space)):
        for j in range(len(scale_space[0])):
            ax[i][j].imshow(scale_space[i][j], cmap='Greys_r')
    plt.show()