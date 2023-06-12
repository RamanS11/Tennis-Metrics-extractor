import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import cv2


# read image into range 0 to 1
ori_img = cv2.imread('results/point_5/frame.png') / 255

# set number of colors
number = 4

# quantize to the amount of colors defined above using kmeans
ori_w, ori_h, _ = ori_img.shape

img = ori_img[ori_w//4:3 * ori_w//4, ori_h//4:3 * ori_h//4, :]
h, w, c = img.shape


def kmeans_clustering(image):
    img2 = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    kmeans_cluster = cluster.KMeans(n_clusters=number)
    kmeans_cluster.fit(img2)
    return kmeans_cluster.cluster_centers_, kmeans_cluster.labels_


# need to scale back to range 0-255 and reshape
cluster_centers, cluster_labels = kmeans_clustering(img)
img3 = (cluster_centers[cluster_labels].reshape(h, w, c) * 255.0).astype('uint8')

cv2.imshow('reduced colors', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# reshape img to 1 column of 3 colors
# -1 means figure out how big it needs to be for that dimension
img4 = img3.reshape(-1, 3)

# get the unique colors
colors, counts = np.unique(img4, return_counts=True, axis=0)
unique = zip(colors, counts)

# need to scale back to range 0-255 and reshape
color = np.where(counts == counts.max())[0][0]


# function to convert from r,g,b to hex
def encode_hex(colour):
    b = colour[0]
    g = colour[1]
    r = colour[2]
    hexa = '#'+str(bytearray([r, g, b]).hex())
    return hexa


# show and save plot (plot each color)
fig = plt.figure()
for i, uni in enumerate(unique):
    color = uni[0]
    count = uni[1]
    plt.bar(i, count, color=encode_hex(color))
plt.show()
fig.savefig('bar_color_histogram.png')
plt.close(fig)


segmented_image = np.zeros_like(ori_img)
segmented_image = np.where()
cv2.imshow('mask', segmented_image)
cv2.waitKey(0)
