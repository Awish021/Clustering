import cv2
import numpy as np
from matplotlib import pyplot as plt

p= '/home/avishay/Project/RawData/Carte Noire other/58184_187646.png'
img = cv2.imread(p)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(3,3,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(sobelx[:,:,0],'Reds')
plt.title('dR/dX'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(sobelx[:,:,1],'Greens')
plt.title('dG/dX'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(sobelx[:,:,2])
plt.title('dB/dX'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,5),plt.imshow(sobely[:,:,0],'Reds')
plt.title('dR/dY'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(sobely[:,:,1],'Greens')
plt.title('dG/dY'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7),plt.imshow(sobely[:,:,2])
plt.title('dB/dY'), plt.xticks([]), plt.yticks([])

plt.show()

plt.imsave('/home/avishay/Project/dB-dY.png',sobely[:,:,2],cmap='Blues')
