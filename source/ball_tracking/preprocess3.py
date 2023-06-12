import os
from glob import glob

import cv2
import numpy as np
import pandas as pd

HEIGHT = 288
WIDTH = 512
mag = 1
sigma = 2.5


def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    heatmap[heatmap <= r ** 2] = 1
    heatmap[heatmap > r ** 2] = 0
    return heatmap * mag


dataset = 'profession_dataset'
# game_list = ['match22']
game_list = ['match1', 'match2', 'match3', 'match4', 'match5', 'match6', 'match7', 'match8', 'match9', 'match10',
             'match11', 'match12', 'match13', 'match14', 'match15', 'match16', 'match17', 'match18', 'match19',
             'match20', 'match21', 'match22', 'match23', 'match24', 'match25', 'match26']
p = os.path.join(dataset, game_list[0], 'frame', '1_01_00', '1.png')
a = cv2.imread(p)
ratio = a.shape[0] / HEIGHT

train_x = []
train_y = []
for game in game_list:
    all_path = glob(os.path.join(dataset, game, 'frame', '*'))
    train_path = all_path[:int(len(all_path) * 1)]
    for i in range(len(train_path)):
        train_path[i] = train_path[i][len(os.path.join(dataset, game, 'frame')) + 1:]
    for p in train_path:
        print(p)
        if not os.path.exists(os.path.join(dataset, game, 'heatmap', p)):
            os.makedirs(os.path.join(dataset, game, 'heatmap', p))
        labelPath = os.path.join(dataset, game, 'ball_trajectory', p + '_ball.csv')
        data = pd.read_csv(labelPath)
        no = data['Frame'].values
        v = data['Visibility'].values
        x = data['X'].values
        y = data['Y'].values
        num = no.shape[0]
        r = os.path.join(dataset, game, 'frame', p)
        r2 = os.path.join(dataset, game, 'heatmap', p)
        x_data_tmp = []
        y_data_tmp = []
        for i in range(num - 2):
            unit = []
            for j in range(3):
                target = str(no[i + j]) + '.png'
                png_path = os.path.join(r, target)
                unit.append(png_path)
            train_x.append(unit)
            unit = []
            for j in range(3):
                target = str(no[i + j]) + '.png'
                heatmap_path = os.path.join(r2, target)
                if v[i + j] == 0:
                    heatmap_img = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
                else:
                    heatmap_img = genHeatMap(WIDTH, HEIGHT, int(x[i + j] / ratio), int(y[i + j] / ratio), sigma, mag)
                heatmap_img *= 255
                unit.append(heatmap_path)
                cv2.imwrite(heatmap_path, heatmap_img)
            train_y.append(unit)

outputfile_name = 'tracknet_train_list_x_3.csv'
with open(outputfile_name, 'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s\n" % (train_x[i][0], train_x[i][1], train_x[i][2]))

outputfile_name = 'tracknet_train_list_y_3.csv'
with open(outputfile_name, 'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s\n" % (train_y[i][0], train_y[i][1], train_y[i][2]))

print('finish')
