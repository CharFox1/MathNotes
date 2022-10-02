# do ocr stuff here to find all chars in image

from matplotlib.ft2font import HORIZONTAL
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import rotate
#from tesseract import image_to_string, image_to_boxes

def showImage(img, name="image"):

    #define the screen resulation
    screen_res = 1000, 600
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    #cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #resize the window according to the screen resolution
    cv2.resizeWindow(name, window_width, window_height)

    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readImage(filename="images/puppy.jpg"):
    # Load an color image in color
    img = cv2.imread(filename,1)

    # show image
    showImage(img)
    return img

def find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skewCorrect(img):

    # grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showImage(img_grey,"grayscale")

    # thresholding
    imgT = cv2.adaptiveThreshold(img_grey,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,3)
    #imgT = cv2.threshold(img_grey, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    showImage(imgT, "threshold")

    #imgT = cv2.fastNlMeansDenoising(imgT, None, 7, 21, 12)
    #showImage(imgT, "denoise")

    # canny edge detection
    #imgT = cv2.bitwise_not(cv2.Canny(img, 50, 200))
    """
    Z = img_grey.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    initLabels = np.array([[0], [255]])
    ret,label,center=cv2.kmeans(Z,K,initLabels,criteria,10,cv2.KMEANS_USE_INITIAL_LABELS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    imgK = res.reshape((img_grey.shape))
    """
    #showImage(imgK, "k-means")

    # erosion and dilation 
    # flip black and white
    imgT = cv2.bitwise_not(imgT, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgT = cv2.erode(imgT, kernel, iterations=2)
    imgT = cv2.morphologyEx(imgT, cv2.MORPH_OPEN, se1, iterations=2)
    showImage(imgT, "open")
    #imgT = cv2.erode(imgT, se1, iterations=1)
    #imgT = cv2.erode(imgT, se2, iterations=1)
    imgT = cv2.morphologyEx(imgT, cv2.MORPH_CLOSE, se1, iterations=2)
    showImage(imgT, "close")
    imgT = cv2.dilate(imgT, kernel, iterations=5)

    # flip black and white back
    imgT = cv2.bitwise_not(imgT, 1)

    showImage(imgT, "eroded")
    
    delta = 1
    limit = 30
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(imgT, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.format(best_angle))
    # correct skew
    data = rotate(imgT, best_angle, reshape=False, order=0, cval=255)
    return data

def preprocess(img):

    # grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showImage(img_grey,"grayscale")

    # blur
    img_blur = cv2.blur(img_grey, ksize=(3,3))
    showImage(img_blur,"blur")
    
    # skew correction
    img_skew = skewCorrect(img)
    showImage(img_skew,"skew")

    # denoise
    dst = cv2.fastNlMeansDenoising(img_skew, None, 10, 10, 12)
    showImage(dst, "denoise")

    return dst

def findLines(img):

    # flip black and white back
    #img = cv2.bitwise_not(img, 1)

    #horizontal_hist = np.sum(img,axis=1,keepdims=True)/255
    #print(horizontal_hist.shape)
    #print(max(horizontal_hist))
    #plt.barh(range(1531), 1, horizontal_hist)
    #plt.show()
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont = cv2.drawContours(img, contours, -1, (200,0,200), 3)
    showImage(cont, "contours")

    return img

def main():
    #img = readImage("images/notes3.png")
    #img = readImage("images/puppy.jpg")
    img = readImage("images/helloWorldEasy.jpg")
    #img = readImage("images/twoLinesRotated.jpg")

    #print(image_to_boxes("images/helloWorldEasy.jpg"))

    img = preprocess(img)
    findLines(img)

if __name__ == "__main__":
    main()