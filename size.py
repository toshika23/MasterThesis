#from PIL import Image
#im=Image.open("sompic.png")
#print im.size 


import PIL
from PIL import Image

basewidth = 12
img = Image.open('img_0063_abs.png')
print img.size
#wpercent = (basewidth/float(img.size[0]))
#hsize = int((float(img.size[1])*float(wpercent)))
hsize = 12
img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
img.save('sompic.png')
print img.size