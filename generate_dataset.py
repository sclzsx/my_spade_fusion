from pathlib import Path
import shutil
import os
import cv2
import numpy as np
from skimage import io
import numpy as np
import random


def mkdir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def generate_dark_and_resave(dir, out):
    for path in Path(dir).rglob('*.tiff'):
        print('processing', str(path))
        parent = path.parent
        parent = str(parent).split('/')[2]
        # print(parent)
        name = path.name
        new_name = parent + '_' + name.split('_')[0] + '.tiff'
        if 'rgb.tiff' in name:
            dir_ = out + '/rgb/'
            mkdir(dir_)
            print(dir_ + new_name)
            shutil.copy(str(path), dir_ + new_name)

            dir_ = out + '/dark/'
            mkdir(dir_)
            img = cv2.imread(str(path)).astype('float32') / 255.0
            gamma = 2
            dark = ((img ** gamma) * 255.0).astype('uint8')
            print(dir_ + new_name)
            cv2.imwrite(dir_ + new_name, dark)

        elif 'nir.tiff' in name:
            dir_ = out + '/nir/'
            mkdir(dir_)
            print(dir_ + new_name)
            shutil.copy(str(path), dir_ + new_name)

def image2patches_random(img1, img2, img3, win, num):
    assert img1.shape == img2.shape
    assert img3.shape == img2.shape
    h, w, _ = img1.shape
    assert win < h and win < w
    
    patch_pairs = []
    n = 0
    while True:
        rand_tl_x = random.randint(0, w - win)
        rand_tl_y = random.randint(0, h - win)
        br_y = rand_tl_y + win
        br_x = rand_tl_x + win

        if br_x < w and br_y < h:
            rand_patch1 = img1[rand_tl_y: br_y, rand_tl_x:br_x]
            rand_patch2 = img2[rand_tl_y: br_y, rand_tl_x:br_x]
            rand_patch3 = img3[rand_tl_y: br_y, rand_tl_x:br_x]
            patch_pairs.append((rand_patch1, rand_patch2, rand_patch3))
            n += 1
            if n >= num:
                break
    return patch_pairs

def crop_and_resave(out, train_rate, patch_size, patch_num):
    nirs = [i for i in Path(out + '/nir').glob('*.*')]
    random.shuffle(nirs)

    num = len(nirs)
    train_num = int(num * train_rate)
    train_nirs = nirs[:train_num]
    test_nirs = nirs[train_num:]
    print('total_num:{}, train_num:{}'.format(num, train_num))

    mkdir(out + '/train/nir')
    mkdir(out + '/train/rgb')
    mkdir(out + '/train/dark')
    mkdir(out + '/test/nir')
    mkdir(out + '/test/rgb')
    mkdir(out + '/test/dark')

    for nir in train_nirs:
        print('processing', nir.name)
        path1 = str(nir)
        path2 = str(nir).replace('nir', 'rgb')
        path3 = str(nir).replace('nir', 'dark')

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img3 = cv2.imread(path3)

        pairs = image2patches_random(img1, img2, img3, patch_size, patch_num)

        for i, pair in enumerate(pairs):
            dst1, dst2, dst3 = pair[0], pair[1], pair[2]

            cv2.imwrite(out + '/train/nir/' + nir.name[:-5] + '_' + str(i) + '.jpg', dst1)
            cv2.imwrite(out + '/train/rgb/' + nir.name[:-5] + '_' + str(i) + '.jpg', dst2)
            cv2.imwrite(out + '/train/dark/' + nir.name[:-5] + '_' + str(i) + '.jpg', dst3)
            # print(dst1.shape, dst2.shape, dst3.shape)

    for nir in test_nirs:
        print('processing', nir.name)
        path1 = str(nir)
        path2 = str(nir).replace('nir', 'rgb')
        path3 = str(nir).replace('nir', 'dark')

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img3 = cv2.imread(path3)

        cv2.imwrite(out + '/test/nir/' + nir.name[:-5] + '_' + str(i) + '.jpg', img1)
        cv2.imwrite(out + '/test/rgb/' + nir.name[:-5] + '_' + str(i) + '.jpg', img2)
        cv2.imwrite(out + '/test/dark/' + nir.name[:-5] + '_' + str(i) + '.jpg', img3)

if __name__ == '__main__':

    dir = '../nirscene1'
    out = '../dataset'

    train_rate = 0.9
    patch_size = 256
    patch_num = 10

    generate_dark_and_resave(dir, out)

    crop_and_resave(out, train_rate, patch_size, patch_num)