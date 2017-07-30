import os
import xlrd
from DS import Crop
import pylab
import numpy as np
import math
import pickle
import cv2

def get_files_mb_pk():
    path = '/home/avishay/Project/RawData'
    images = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"):
                images.append(os.path.join(root, name))
       # for name in dirs:
        #   print name
    return images

def convert_to_int(item):
    if isinstance(item,basestring):
        return item
    else:
        return int(item)

def create_crops(path,mb_list):
    crops=[]
    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_index(0)
    for row_num in range(sheet.nrows)[1:]:
        row_value = sheet.row_values(row_num)
        mb= str(int(row_value[8]))
        for record in mb_list:
            reversed = record[::-1]
            if mb == (reversed[reversed.find(".")+1:reversed.find("_")])[::-1]:
                url=record
        row_value.append(url)
        row_value.append(url.replace('RawData','BoundingBoxes'))
        row_value=map(convert_to_int,row_value[1::])
        crops.append(Crop(*row_value))
    return crops


def get_circle(im):
    x,y = im.shape[:2]
    return (x/2,y/2,(x+y)/10)


def generate_gaussian_func(mean,sigma):
    def gaussian_func(x):
        var = float(sigma) ** 2
        pi = np.pi
        denom = (2 * pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom
    return gaussian_func

def pixel_value(xn,yn,cx,cy,x,y):
    return math.ceil(xn(x)*cx*yn(y)*cy)


def get_gaussian(im):
    cenx, ceny, rad = get_circle(im)
    xn = generate_gaussian_func(cenx,rad)
    cx = math.sqrt(10)/xn(cenx)
    yn = generate_gaussian_func(ceny,rad)
    cy = math.sqrt(10)/yn(ceny)
    return (xn,yn,cx,cy)

def get_im_array(im):
    arr = []
    sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
    xn,yn,cx,cy= get_gaussian(im)
    pixx,pixy=im.shape[:2]
    for x in range(pixx):
        for y in range(pixy):
            lst = [x,y]
            lst.extend(im[x, y])
            lst.extend(sobelx[x, y])
            lst.extend(sobely[x, y])

            times = int( pixel_value(xn,yn,cx,cy,x,y))
            arr.extend([lst]*times)
    return np.asarray(arr)

def image_to_matrix(im):
    arr = get_im_array(im)
    arr.astype(float)
    xs = arr[:,0].ravel()
    ys = arr[:,1].ravel()
    r = arr[:,2].ravel()
    g = arr[:,3].ravel()
    b = arr[:,4].ravel()
    # drdx= arr[:,5].ravel()
    # dgdx = arr[:, 6].ravel()
    # dbdx = arr[:, 7].ravel()
    # drdy = arr[:, 8].ravel()
    # dgdy = arr[:, 9].ravel()
    # dbdy = arr[:, 10].ravel()
    #
    # f= np.vstack([xs,ys,r,g,b,drdx,dgdx,dbdx,drdy,dgdy,dbdy])
    #
    # f = np.dot(np.diag([1,1,3,3,3,5,5,5,5,5,5]),f)
    f = np.vstack([xs,ys,r,g,b])
    S = np.cov(f)
    return S

def image_to_vector(im):
    mat = image_to_matrix(im)
    arr = []
    for x in range(mat.shape[1]):
        for y in range(mat.shape[1]):
            if y<=x:
                arr.extend([mat[x,y]])
    return np.asarray(arr)


# def createBoundedBoxes(Objs):
#     count = 0
#     for obj in objs:
#         count=count+1
#
#         im =  pylab.imread(obj.url)
#         x,y=im.shape[:2]
#
#         print im.shape
#         print str(x) + ',' + str(y) +','+obj.url
#         boundedBox=im[x/3:2*x/3,y/4:3*y/4,:]
#         pylab.imsave(obj.bburl,boundedBox)
# createBoundedBoxes(objs)

objs = create_crops('/home/avishay/Project/crop_metadata.xlsx',get_files_mb_pk())

sum =0

path = '/home/avishay/Project/pickle-files/'
for obj in objs:
    sum=sum+1
    print sum
    im = cv2.imread(obj.url)
    obj.data=image_to_vector(im).tolist()
    filename = path + str(obj.probe_fk) + "_" + str(obj.mb_pk) + ".p"
    pickle.dump(obj, open(filename, "wb"))

# im = pylab.imread(objs[149].url)
#
# cenx,ceny,rad = get_circle(im)
# x,y = im.shape[:2]
# xn,yn,cx,cy = get_gaussian(im)
# print pixel_value(xn,yn,cx,cy,0,0)
#
# arr = image_to_vector(im)
#
