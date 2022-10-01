# do ocr stuff here to find all chars in image

import numpy as np
import cv2

def readImage():
    # Load an color image in color
    img = cv2.imread("images/puppy.jpg",1)
    
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def main():
    img = readImage()

if __name__ == "__main__":
    main()