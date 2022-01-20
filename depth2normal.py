import cv2
import numpy as np
from matplotlib import pyplot as plt
d_im = cv2.imread("D:\\Replica_scenes\\frl_apartment_0\\test.depth.00800.png")
d_im = d_im.astype("float64")
d_im = d_im[:, :, 1]
h, w = np.shape(d_im)
normals = np.zeros((h, w, 3))

def normalizeVector(v):
    length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    v = v/length
    return v

for x in range(1, h-1):
    for y in range(1, w-1):

        dzdx = (float((d_im[x+1, y])) - float((d_im[x-1, y]))) / 2.0
        dzdy = (float((d_im[x, y+1])) - float((d_im[x, y-1]))) / 2.0

        d = (-dzdx, -dzdy, 1.0)

        n = normalizeVector(d)

        normals[x,y] = n * 0.5 + 0.5

normals *= 255

normals = normals.astype('uint8')
# plt.imshow(normals)
# plt.show()
normals = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)

cv2.imwrite("normal.png", normals)
