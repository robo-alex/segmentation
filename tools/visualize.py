import numpy as np
from PIL import Image

data = np.load('D:\\1_12\\frl2_0112\\000_normal.npy')
data = data * 255
# r=Image.fromarray(data[0]).convert("L")
# g=Image.fromarray(data[1]).convert("L")
# b=Image.fromarray(data[2]).convert("L")

# img=Image.merge("RGB",(r,g,b))
image = Image.fromarray(np.uint8(data))
# image.save('1448291769966919811.jpg')
image.save('D:\\1_12\\frl2_0112\\000_normal.jpg')
