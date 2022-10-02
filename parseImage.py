# do ocr stuff here to find all chars in image

import numpy as np
import cv2

def showImage(img, name="image"):
    cv2.imshow(name,img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def readImage(filename="images/puppy.jpg"):
    # Load an color image in color
    img = cv2.imread(filename,1)
    
    # resize image
    #scale_percent = 30 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
  
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    img = cv2.resize(img, (500, 1000))

    # show image
    showImage(img)
    return img

def preprocess(img):

    # grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showImage(img_grey,"grayscale")

    # thresholding
    imgf = cv2.adaptiveThreshold(img_grey,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,20)
    showImage(imgf, "threshold")

    
    # skew correction


    # denoise
    dst = cv2.fastNlMeansDenoising(imgf, None, 10, 10, 12)
    showImage(dst, "denoise")

    return imgf

def findLines(img):

    
    
    return img

def main():
    img = readImage("images/notes1.png")
    preprocess(img)

if __name__ == "__main__":
    main()