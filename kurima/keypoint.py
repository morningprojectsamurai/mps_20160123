from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace
from dog import dog


def extract_keypoints(octave, threshold):
    ## ここに処理を書く
    keypoints = []

    return keypoints


if __name__ == '__main__':
    lena_img = Image.open('img/char.jpg').convert('L')
    lena_img = np.array(lena_img, dtype=np.float) / 255

    print('Create Scale-space')
    scale_space = ScaleSpace(lena_img)
    scale_space.create()

    print('Apply DoG to Scale-space')
    dog_space = dog(scale_space)

    keypoint_space = []
    for octave in dog_space:
        keypoint_space.append([])
        keypoint_space[-1].append(extract_keypoints(octave, 0))

    print('Draw keypoints')
    plt.imshow(lena_img, cmap='Greys_r')
    fig = plt.gcf()
    for n, octave in enumerate(keypoint_space):
        r = 1
        for l, layer in enumerate(octave):
            print(len(layer))
            for p in layer:
                p *= np.power(2, n)
                fig.gca().add_artist(plt.Circle((p[1], p[0]), r, color='r', fill=False))
    plt.show()