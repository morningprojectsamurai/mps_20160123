from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace
from dog import dog


def extract_keypoints(octave, threshold):
    ## ここに処理を書く
    keypoints = []

    for k in range(1,len(octave)-1):
        h,w = octave[k].shape
        for i in range(1,h-1):
            for j in range(1,w-1):

                # ターゲットの値を取得
                v = octave[k][i][j]  # ターゲットの値

                # threshold以下の場合はスキップ
                if v < threshold: continue

                # 前後の画像の同一座標上の値を取得
                v_prev = octave[k-1][i][j]  # 1つ前の画像の値
                v_next = octave[k+1][i][j]  # 1つ後の画像の値

                # i-1からi+1, j-1からj+1の9個のセルを取得
                rng = octave[k][i-1:i+2,j-1:j+2]

                # 極大を判定
                if isLocalMax(v, rng) and v > np.max(v_prev, v_next):
                    keypoints.append([i,j])

                # 極小を判定
                if isLocalMin(v, rng) and v < np.min(v_prev, v_next):
                    keypoints.append([i,j])

    return keypoints

# (画像内の)極大値かを判定
def isLocalMax(target, rng):
    r = rng
    r[1][1] = -1 # 自分自身はmaxから除外する
    max_value = np.max(r)
    if target > max_value:
        return True
    else:
        return False

# (画像内の)極小値かを判定
def isLocalMin(target, rng):
    r = rng
    r[1][1] = 101 # 自分自身はminから除外する
    min_value = np.min(r)
    if target < min_value:
        return True
    else:
        return False

if __name__ == '__main__':
    lena_img = Image.open('img/lena.png').convert('L')
    lena_img = np.array(lena_img, dtype=np.float) / 255

    print('Create Scale-space')
    scale_space = ScaleSpace(lena_img)
    scale_space.create()

    print('Apply DoG to Scale-space')
    dog_space = dog(scale_space)

    keypoint_space = []
    for octave in dog_space:
        keypoint_space.append([])
        keypoint_space[-1].append(extract_keypoints(octave, 0.0003))

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