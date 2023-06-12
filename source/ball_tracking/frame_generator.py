import os
from glob import glob

import cv2

game_list = ['match1','match2','match3','match4','match5','match6','match7','match8','match9','match10','match11','match12','match13','match14','match15','match16','match17','match18','match19','match20','match21','match22','match23','match24','match25','match26']
for game in game_list:
	p = os.path.join('profession_dataset',game, 'rally_video', '*mp4')
	video_list = glob(p)
	os.makedirs(game + '/frame/')
	for videoName in video_list:
		rallyName = videoName[len(os.path.join(game, 'rally_video'))+1:-4]
		outputPath = os.path.join(game, 'frame', rallyName)
		outputPath += '/'
		os.makedirs(outputPath)
		cap = cv2.VideoCapture(videoName)
		success, count = True, 0
		success, image = cap.read()
		while success:
			cv2.imwrite(outputPath + '%d.png' %(count), image)
			count += 1
			success, image = cap.read()

