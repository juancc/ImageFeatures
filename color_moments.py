import cv2
from scipy.stats import skew, kurtosis 

# BGR
blue = cv2.imread('Assets/nino.jpeg')[:,:,0].flatten()

sk = skew(blue, axis=0, bias=True)
kt = kurtosis(blue, axis=0, bias=True)

print(f'{sk=}, {kt=}')