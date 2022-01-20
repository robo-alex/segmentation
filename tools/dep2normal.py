import cv2
import numpy as np
d_im = cv2.imread("D:\\Replica_scenes\\frl_apartment_0\\test.depth.00800.png", cv2.IMREAD_UNCHANGED)
cv2.imwrite("depth.jpg", d_im)
print(d_im.shape)
print(d_im)
# d_im = d_im.astype("float64")
# d_im = d_im[:, :, 1]
print(d_im.shape)

# normals = np.array(d_im, dtype="float32")
# h,w,d = d_im.shape
# for i in range(1,w-1):
#   for j in range(1,h-1):
#     t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
#     f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
#     c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
#     d = np.cross(f-c,t-c)
#     n = d / np.sqrt((np.sum(d**2)))
#     normals[j,i,:] = n

# cv2.imwrite("normal.jpg",normals*255)
zy, zx = np.gradient(d_im)
# You may also consider using Sobel to get a joint Gaussian smoothing and differentation to reduce noise
# zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
# zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
n = np.linalg.norm(normal, axis=2)
normal[:, :, 0] /= n
normal[:, :, 1] /= n
normal[:, :, 2] /= n

# offset and rescale values to be in 0-255
normal += 1
normal /= 2
normal *= 255
# print(normal.shape)
# normal = cv2.cvtColor(normal[:, :, ::-1], cv2.COLOR_RGB2BGR)

cv2.imwrite("normal.png", normal[:, :, ::-1])
