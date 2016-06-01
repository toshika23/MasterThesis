import timeit


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL
import cv2
import os  


def improveimages(images,labels):
    print "hello"
    indir1 = '/home/gordon/imagesmoothing/trainimages'
    indir2 = '/home/gordon/imagesmoothing/trainlabels'
    
    basewidth = 561
    hsize = 427
    suffix1 = '.jpg'
    a1=os.path.join(images + suffix1)
    
    suffix2 = '.png'
    a2=os.path.join(images + suffix2)

 
  
    img_depth = Image.open(a2)
    img_depth = img_depth.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    pix_depth = img_depth.load()

    basewidth = 561
    hsize = 427
    img_gray = Image.open(a1)
    img_gray = img_gray.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    pix_gray = img_gray.load()
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0
    h=0

    width_depth =img_depth.size[0]
    height_depth = img_depth.size[1] 

    for threshold in [12,25,50,75,255]:
            print "hello"
            for i in range(1,width_depth-1):
                for j in range(1,height_depth-1):
                    a=abs(pix_gray[i,j]-pix_gray[i-1,j-1])
                    b=abs(pix_gray[i,j]-pix_gray[i,j-1])
                    c=abs(pix_gray[i,j]-pix_gray[i+1,j-1])
                    d=abs(pix_gray[i,j]-pix_gray[i-1,j])
                    e=abs(pix_gray[i,j]-pix_gray[i+1,j])
                    f=abs(pix_gray[i,j]-pix_gray[i-1,j+1])
                    g=abs(pix_gray[i,j]-pix_gray[i,j+1])
                    h=abs(pix_gray[i,j]-pix_gray[i+1,j+1])
                    if pix_depth[i,j]==0:
                        
                        if (a<threshold):
                            avg1= pix_depth[i-1,j-1]
                        
                        else:
                            avg1=0
                        
                        if (b<threshold):
                            avg2=pix_depth[i,j-1]
                        else:
                            avg2=0
                        if (c<threshold):
                            avg3=pix_depth[i+1,j-1]
                        else:
                            avg3=0
                        if (d<threshold):
                            avg4=pix_depth[i-1,j]
                        else:
                            avg4=0
                        if (e<threshold):
                            avg5=pix_depth[i+1,j]
                        else:
                            avg5=0
                        if (f<threshold):
                            avg6=pix_depth[i-1,j+1]
                        else:
                            avg6=0
                        if (g<threshold):
                            avg7=pix_depth[i,j+1]
                        else:
                            avg7=0
                        if (h<threshold):
                            avg8=pix_depth[i+1,j+1]
                        else:
                            avg8=0;
                        ain= np.array([[avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8]])
                        
                        x=(ain != 0).sum(1)
                        if x!=0:
                            aou=ain.sum(1)/x
                            pix_depth[i,j]=aou[0]
                    #print pix_depth[i,j]
                        
    img_depth.save(a2) 
   

if __name__ == '__main__':
    start = timeit.default_timer()
    images=[]
    labels=[]
    indir1 = '/home/gordon/imagesmoothing/trainimages'
    indir2 = '/home/gordon/imagesmoothing/trainlabels'
    for root, dirs, filenames in os.walk(indir1):
        for f in filenames:
            #print(f)
            extension1 = os.path.splitext(f)[0]
            images.append(extension1)
    for root, dirs, filenames in os.walk(indir2):
        for f in filenames:
            #print(f)
            extension2 = os.path.splitext(f)[0]
            #images.append(extension2)
            labels.append(extension2)
    #count=0
    for i in range(0,len(images)-1):
        for j in range(0,len(labels)-1):
            if images[i]==labels[j]:
                print "hello"
                improveimages(images[i],labels[j])
        
    #print count
    stop = timeit.default_timer()

    print stop - start 