# encoding:utf-8

import os

# Path of Files
path = '../images_sample'
src_path = path
des_path = path
train_out_file = des_path + '/train.txt'
val_out_file = des_path + '/val.txt'

train_out = open(train_out_file, 'w')
val_out = open(val_out_file, 'w')

for root, dirs, files in os.walk(src_path, topdown=False):
    for name in files:
        # 判断是否是图片文件
        if '.jpg' not in name:
            continue
        if 'train' in root:
            if 'nonrumor' in root:
                train_out.write(name + ' 0\n')
            else:
                train_out.write(name + ' 1\n')
        if 'validation' in root:
            if 'nonrumor' in root:
                val_out.write(name + ' 0\n')
            else:
                val_out.write(name + ' 1\n')

train_out.close()
val_out.close()