from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace
from dog import dog


def extract_keypoints(octave, threshold):
    ## ここに処理を書く
    keypoints = []
    for i in range(1, octave.shape[0]-1):
        for x in range(1, octave[i].shape[0]-1):
            for y in range(1, octave[i].shape[1]-1):
                v = [octave[i+a][x+b][y+c] for a in [-1, 0, 1]
                                           for b in [-1, 0 ,1]
                                           for c in [-1, 0 ,1]
                ]
                check_v = octave[i][x][y]
                max_v = max(v)
                min_v = min(v)
                diff_v = max_v - min_v
                if check_v in (max_v, min_v) and diff_v >= threshold:
                    keypoints.append((x, y, diff_v))

    return np.array(keypoints)


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
        keypoint_space[-1].append(extract_keypoints(octave, 0.005))

    print('Draw keypoints')
    plt.imshow(lena_img, cmap='Greys_r')
    fig = plt.gcf()
    for n, octave in enumerate(keypoint_space):
        for l, layer in enumerate(octave):
            print(len(layer))
            for p in layer:
                p[:2] *= np.power(2, n)
                fig.gca().add_artist(plt.Circle((p[1], p[0]), p[2]*300, color='r', fill=False))
    plt.show()