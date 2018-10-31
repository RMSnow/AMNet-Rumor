# encoding:utf-8

from urllib.request import urlretrieve
import json
import socket
import time

with open('url_json.txt', 'r') as src:
    lines = src.readlines()
    pic_lists = []
    for json_obj in lines:
        url_dict = json.loads(json_obj)
        if 'piclists' in url_dict.keys():
            curr_pic = url_dict['piclists']
            pic_lists += curr_pic

pic_set = set(pic_lists)

# 数量统计
print('-------------------------')
print('There are {} imgs in pic_lists.'.format(len(pic_lists)))
print('Unique imgs are {}.'.format(len(pic_set)))
print()

start_time = time.time()
# 设置超时时间为30s
socket.setdefaulttimeout(10)

size = len(pic_set)
i = 0
for pic_url in pic_set:
    pic_name = pic_url.split('/')[-1]
    try:
        urlretrieve(pic_url, '/media/Data/qipeng/modified_complete_images/crawler/' + pic_name)
    except socket.timeout:
        count = 1
        while count <= 5:
            try:
                urlretrieve(pic_url, pic_name)
                break
            except socket.timeout:
                err_info = 'Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count
                print(err_info)
                count += 1
        if count > 5:
            print("downloading picture fialed!")
    except:
        print("[Error] Something wrong in downloading {} !".format(pic_url))
        print()

    i += 1
    if i % 50 == 0:
        print('Downloading {} pics, {:.2f}%, {:.2f} sec...'.format(i, i / size * 100, time.time() - start_time))

print('-------------------------')
