import cv2
import numpy as np

print("ENTER ANY KEY TO SEE NEXT IMAGE")
#output 1

def colourchange(img):
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    cv2.imshow("original",img)
    cv2.imshow("output", imgHSV)
    cv2.waitKey(0)

img1=cv2.imread('1.png')
colourchange(img1)

img2=cv2.imread('2.png')
colourchange(img2)

img3=cv2.imread('3.png')
colourchange(img3)

img4=cv2.imread('4.png')
colourchange(img4)

img5=cv2.imread('5.png')
colourchange(img5)

img6=cv2.imread('6.png')
colourchange(img6)

img7=cv2.imread('7.png')
colourchange(img7)

img8=cv2.imread('8.png')
colourchange(img8)

img9=cv2.imread('9.png')
colourchange(img9)

img10=cv2.imread('10.png')
colourchange(img10)

#output3
myColors = [[1,179,0,255,0,255],
            [60,179,0,255,0,255]]
