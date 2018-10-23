# encoding:utf-8

import pandas as pd

pred_src_file = 'pred.txt'
pred_real_file = 'pred-real-sorted.txt'


def get_pred_label(threshold):
    des_file = 'pred-threshold' + str(threshold) + '.txt'
    with open(pred_src_file, 'r') as src:
        with open(des_file, 'w') as des:
            src_lines = src.readlines()
            for src_line in src_lines:
                img_name, pred_value = src_line.split()
                if float(pred_value) > threshold:
                    des.write(img_name + ' 1\n')
                else:
                    des.write(img_name + ' 0\n')


# 阈值为0.5的Accuracy
# get_accuracy(0.5)

# 合并true标签与pred标签的两个文件
# def merge():
#     des_file = 'eval.csv'
#     with open('pred-threshold0.5.txt', 'r') as y_pred_file:
#         with open('pred-real-sorted.txt', 'r') as y_true_file:
#             with open(des_file, 'w') as des:
#                 y_pred_file_lines = y_pred_file.readlines()
#                 y_true_file_lines = y_true_file.readlines()
#                 for y_pred_file_line in y_pred_file_lines:


