# encoding:utf-8

src_file = 'pred-real.txt'
des_file = 'pred-real-sorted.txt'

with open(src_file, 'r') as src:
    with open(des_file, 'w') as des:
        src_lines = src.readlines()
        src_lines.sort()
        for src_line in src_lines:
            des.write(src_line)