# do ocr stuff here to find all chars in image

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread("images/puppy.jpg",0)
 
# show image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()