# encoding:utf-8

import os

for i in range(4):
    expt_name = str(i + 1) + '.2'
    os.system('mkdir /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt%s' % (expt_name))
    os.system('mkdir /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt%s/att_maps' % (expt_name))
    os.system('mkdir /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt%s/att_maps/rumor' %
              (expt_name))
    os.system('mkdir /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt%s/att_maps/nonrumor' %
              (expt_name))