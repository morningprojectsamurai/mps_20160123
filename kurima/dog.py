from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace


def dog(scale_space):
    doged_space = []
    for octave in scale_space:
        doged_space.append([])
        for l in range(1, len(octave)):
            doged_space[-1].append(octave[l] - octave[l - 1])
    return np.array(doged_space)


if __name__ == '__main__':
    lena_img = Image.open('img/lena.jpg').convert('L')
    lena_img = np.array(lena_img, dtype=np.float) / 255

    print('Create Scale-space')
    scale_space = ScaleSpace(lena_img)
    scale_space.create()

    print('Apply DoG to Scale-space')
    dog_space = dog(scale_space)

    print('Draw DoG-space')
    f, ax = plt.subplots(len(dog_space), len(dog_space[0]))
    for i in range(len(dog_space)):
        for j in range(len(dog_space[0])):
            ax[i][j].imshow(dog_space[i][j], cmap='Greys_r')
    plt.tight_layout()
    plt.savefig('img/goded_space.png')
    plt.show()