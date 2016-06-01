import sys
import os
import numpy as np
import cv2
 
def greyscale(img):
  '''Convert an image to greyscale.
  image  - a numpy array of shape (rows, columns, 3).
  output - a grey scale image which is a numpy array of shape (rows, columns) 
           containing the average of image's 3 channels. 
  '''
  image = np.uint16(img)
  avg = np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  avg[:, :,0] = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2])/3
  output = avg.astype(np.uint8)
  return output[:,:,0]
 
def main():
  '''Convert images to greyscale.
 
  searches for images in directory images/knowapa, and applies the grey scale 
  functiony to each image in the same directory and saves greyscaleimage for 
  each file with the word grey appended to the image
  '''
 
  imagesfolder = os.path.join('/home/gordon/TrainingData/data1/jpgonlydata/')
  exts = ['.jpg']
  for dirname, dirnames, filenames in os.walk(imagesfolder):
    for filename in filenames:
      name, ext = os.path.splitext(filename)
      
      if ext in exts and 'grey' not in name:
        img = cv2.imread(os.path.join(dirname, filename))
        #print img
        greyimage = greyscale(img)
        cv2.imwrite(os.path.join(dirname, name+ext), greyimage)
 
 
 
if __name__ == "__main__":
  main()