import pandas

import shutil


HEIGHT, WIDTH = 720, 1280

mode = 'test'

a = pandas.read_csv(f'signs/rtsd-d3-gt/blue_rect/{mode}_gt.csv')
path = '/home/azarechny/university/OIvIS/3/data'
length = a.shape[0]

num = 0
classes = {'5_20': 0, '5_19_1': 1, '5_15_5': 2, '6_3_1': 3, '6_7': 4, '5_15_3': 5, '6_4': 6, '6_6': 7, '5_15_1': 8, '5_15_2': 9, '5_6': 10, '5_5': 11, '5_15_2_2': 12, '5_22': 13, '5_3': 14, '6_2_n50': 15, '6_2_n70': 16, '5_15_7': 17, '5_14': 18, '5_21': 19, '6_2_n60': 20, '5_7_1': 21, '5_7_2': 22, '5_11': 23, '5_8': 24}


for index in range(length):

    item = a.iloc[[index]]

    filename = item['filename'].values[0]

    number = item['sign_class'].values[0]

    height = item['height'].values[0]
    width = item['width'].values[0]

    center_x = item['x_from'].values[0] + height / 2
    center_y = item['y_from'].values[0] + width / 2

    center_x /= WIDTH
    width /= WIDTH

    center_y /= HEIGHT
    height /= HEIGHT

    # if classes.get(number, None) is None:
    #     classes[number] = num
    #     num += 1
    if classes.get(number, None) is None:
        continue

    shutil.copyfile(f'{path}/signs/rtsd-d3-frames/{mode}/{filename}', f'{path}/dataset/images/{mode}/{filename}')

    with open(f'dataset/labels/{mode}/{filename.split('.')[0]}.txt', 'w+') as file:
        file.write(f'{classes[number]} {center_x:.5f} {center_y:.5f} {width:.5f} {height:.5f}')

    print(f'{classes[number]} {center_x:.5f} {center_y:.5f} {width:.5f} {height:.5f}')


# num = 0
# classes = {}
#
# for index in range(length):
#
#     item = a.iloc[[index]]
#
#     filename = item['filename'].values[0]
#
#     # number = item['sign_class'].values[0]
#     #
#     # height = item['height'].values[0]
#     # width = item['width'].values[0]
#     #
#     # center_x = item['x_from'].values[0] + height / 2
#     # center_y = item['y_from'].values[0] + width / 2
#     #
#     # center_x /= WIDTH
#     # width /= WIDTH
#     #
#     # center_y /= HEIGHT
#     # height /= HEIGHT
#     #
#     # if classes.get(number, None) is None:
#     #     classes[number] = num
#     #     num += 1
#
#
#
#
#     # with open(f'dataset/labels/{mode}/{filename.split('.')[0]}.txt', 'w+') as file:
#     #     file.write(f'{classes[number]} {center_x:.5f} {center_y:.5f} {width:.5f} {height:.5f}')
#

print(classes)
