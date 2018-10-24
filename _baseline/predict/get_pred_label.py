# encoding:utf-8

import os

# Path of Files
src_path = '/media/Data/qipeng/modified_complete_images'
des_path = '/media/Data/qipeng/modified_complete_images/pre_handle_add_label'
val_out_file = des_path + '/pred.txt'

val_out = open(val_out_file, 'w')

for root, dirs, files in os.walk(src_path, topdown=False):
    for name in files:
        # 判断是否是图片文件
        # if '.jpg' not in name:
        #     continue
        if 'validation' in root:
            if 'nonrumor' in root:
                val_out.write(name + ' 0\n')
            else:
                val_out.write(name + ' 1\n')

val_out.close()
