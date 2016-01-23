#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace
from dog import dog


def get_extreme_value(img, r, c):
    extreme = [256, -1]
    for i in range(-1,2,1):
        for j in range(-1,2,1):
        # get minimum
            if img[r+i][c+j] < extreme[0]:
                extreme[0] = img[r+i][c+j]
        # get maximum
            if img[r+i][c+j] > extreme[1]:
                extreme[1] = img[r+i][c+j]
            
    return extreme

def get_keypoints(curimg, previmg, nextimg, keypoints, threshold):
    rows = curimg.shape[0]
    cols = curimg.shape[1]
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            minv, maxv = get_extreme_value(curimg, r, c)
            if maxv - minv <= threshold:
                continue
            center = curimg[r][c]
            if center != minv and center != maxv:
                continue

            minv, maxv = get_extreme_value(previmg, r, c)
            if center > minv and center < maxv:
                continue

            minv, maxv = get_extreme_value(nextimg, r, c)
            if center > minv and center < maxv:
                continue
            keypoints.append([r,c])


def extract_keypoints(octave, threshold):
    ## ここに処理を書く
    keypoints = []

    # don't deal with first element and last element
    end = len(octave) - 1
    for i in range(1, end):
        img = octave[i]
        previmg = octave[i-1]
        nextimg = octave[i+1]
        get_keypoints(img, previmg, nextimg, keypoints, threshold)

    return keypoints


if __name__ == '__main__':
    lena_img = Image.open('img/lena.jpg').convert('L')
    lena_img = np.array(lena_img, dtype=np.float) / 255.0

    print('Create Scale-space')
    scale_space = ScaleSpace(lena_img)
    scale_space.create()

    print('Apply DoG to Scale-space')
    dog_space = dog(scale_space)

    keypoint_space = []
    for octave in dog_space:
        keypoint_space.append([])
        keypoint_space[-1].append(extract_keypoints(octave, 0.003))

    print('Draw keypoints')
    plt.imshow(lena_img, cmap='Greys_r')
    fig = plt.gcf()
    for n, octave in enumerate(keypoint_space):
        r = 1
        for l, layer in enumerate(octave):
            print(len(layer))
            for p in layer:
                p *= np.power(2.0, n)
                fig.gca().add_artist(plt.Circle((p[1], p[0]), r, color='r', fill=False))
    plt.show()
