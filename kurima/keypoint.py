from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scale_space import ScaleSpace
from dog import dog


def extract_keypoints(octave, threshold):
    
    TH = 0.01
    
    # キーポイントの x, y, z の配列の配列
    resultpoints = []

    for z in range(len(octave)):
        for y in range(len(octave[0])):
            for x in range(len(octave[0][0])):
                
                is_min = True
                is_max = True
                
                src_value = octave[z][y][x]
                
                # 中途半端な値は無視する
                if src_value < TH:
                    continue
                
                for dz in range(-1, 2):
                    dst_z = z + dz                    
                    # 範囲チェック
                    if dst_z < 0 or dst_z >= len(octave):
                        continue

                    for dy in range(-1, 2):
                        dst_y = y + dy
                        # 範囲チェック
                        if dst_y < 0 or dst_y >= len(octave[0]):
                            continue

                        for dx in range(-1, 2):
                            # 比較対象の位置を得る
                            dst_x = x + dx
                            # 範囲チェック
                            if dst_x < 0 or dst_x >= len(octave[0][0]):
                                continue

                            # 自身とは比較しない
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            
                            if is_max and src_value < octave[dst_z][dst_y][dst_x]:
                                is_max = False
                            if is_min and src_value > octave[dst_z][dst_y][dst_x]:
                                is_min = False
                            
                if is_min or is_max:
                    resultpoints.append([x, y, z])
                                
    keypoints = []
    
    for p in resultpoints:
        keypoints.append([p[1], p[0]])
    

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
                m = np.power(2, n)                
                fig.gca().add_artist(plt.Circle((p[1] * m, p[0] * m), r, color='r', fill=False))                
    plt.show()