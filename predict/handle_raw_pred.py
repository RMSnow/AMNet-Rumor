# encoding:utf-8

raw_file = 'memorabilities.txt'
pred_file = 'pred.txt'

raw_dir_path = '/media/Data/qipeng/modified_complete_images/pre_handle_add_label/AMNet-Predict/images/'

with open(raw_file, 'r') as raw:
    with open(pred_file, 'w') as pred:
        raw_lines = raw.readlines()
        for raw_line in raw_lines:
            line = raw_line.split()
            img_name = line[0].split(raw_dir_path)[1]
            pred_value = line[1]
            pred.write(img_name + ' ' + pred_value + '\n')
