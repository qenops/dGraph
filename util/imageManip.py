#!/usr/bin/python

__author__ = ('David Dunn')
__version__ = '0.1'

import cv2, os, math, sys
sys.path.append(u'C:/Users/qenops/Dropbox/code/python')
import numpy as np

# All measurements are in meters
#pixelDiameter = .000042333  # .0042333cm = 1/600 in (600dpi)
#pixelDiameter = .000333333333  # (3 dpmm)
#pixelDiameter = .0001  # (10 dpmm)

# Draw a string on an image (from cv2 example common.py)
def drawStr(dst, point, s, scale=1.0,thick=1):
    x,y = point
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = thick*2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255),thickness = thick, lineType=cv2.CV_AA)

# Convert a string to a number - only good for 1 number per string (otherwise use regex)
def strToInt(x):
    return int(''.join(ele for ele in x if ele.isdigit()))

def imgToHex(img,fill=2):
    return '\n'.join(['\n'.join([''.join([np.base_repr(i,16).zfill(fill) for i in j]) for j in k]) for k in img])

def fileListToHex(dir,lst):
    myString = ''
    for f in lst:
        img = cv2.cvtColor(cv2.imread(os.path.join(dir,f)),cv2.COLOR_BGR2RGB)
        myString += '%s\n'%imgToHex(img/16,1)
    return myString

def bitmapToImage(myMap,tiles):
    output = None
    for row in myMap:
        curRow = None
        for i in row:
            curRow = np.hstack((curRow,tiles[i])) if curRow is not None else tiles[i]
        output = np.vstack((output, curRow)) if output is not None else curRow
    return output

# Get the scale factor for normalization for numpy dtypes 
def getBitDepthScaleFactor(typeName):
    if typeName[:4] == 'uint':
        return 2**strToNum(typeName)-1
    elif typeName[:3] == 'int':
        return 2**(strToNum(typeName)-1)-1
    else:
        return 1.

# Definition to create circle.
def circle(nx,ny,d,smooth=False,n=1):
    y,x    = np.ogrid[-nx/2: (nx/2), -ny/2: (ny/2)]
    circ = (x+1)**2+(y+1)**2 <= ((d)/2.)**2
    return circ
    
# Get the image in the proper format
def getImageFloat(imgName):
    img = cv2.imread(imgName)
    scale = float(getBitDepthScaleFactor(img.dtype.name))
    img = np.float32(img)/scale
    return img
    
# Get the image in the proper format
def getImageAlpha(imgName):
    img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
    scale = float(getBitDepthScaleFactor(img.dtype.name))
    img = np.float32(img)/scale
    return img
    
# Composite top image over bot image using top's alpha chanel (and bot's alpha chanel) optional
def over(topC, topA, botC, botA=None, premultiply=True):
    if not premultiply:
        topC = cv2.multiply(topC, np.dstack((topA,topA,topA)))
        if botA is not None:
            botC = cv2.multiply(botC, np.dstack((botA,botA,botA)))
    if botA is None:
        botA = np.ones_like(topA)
        outA = botA
    else:
        outA = cv2.add(topA, cv2.multiply(botA, cv2.subtract(1.,topA)))
    outC = cv2.add(topC, cv2.multiply(botC, cv2.subtract(np.ones_like(botC),np.dstack((topA,topA,topA)))))
    return outC, outA

# Calculate the circle of confusion diameter given the focal image depth, blur image depth and aperture diameter
def getCoC(aperture, focalDist, blurDist):
    return aperture * abs(blurDist - focalDist) / blurDist

# Crop or exand (with one padding) the image to the given dimensions
def cropImg(img, refDim):
    size = img.shape[:2]
    if size[0] >= refDim[0] and size[1] >= refDim[1]:
        return img[round(size[0]/2)-round(refDim[0]/2):round(size[0]/2)+refDim[0]-round(refDim[0]/2),round(size[1]/2)-round(refDim[1]/2):round(size[1]/2)+refDim[1]-round(refDim[1]/2)]
    else:
        toRet = np.ones([refDim[0],refDim[1],3], img.dtype.name)
        xMin = min(refDim[0],size[0])
        yMin = min(refDim[1],size[1])
        xMinH = min(round(refDim[0]/2),round(size[0]/2))
        yMinH = min(round(refDim[1]/2),round(size[1]/2))
        toRet[round(refDim[0]/2)-xMinH:round(refDim[0]/2)+xMin-xMinH,round(refDim[1]/2)-yMinH:round(refDim[1]/2)+yMin-yMinH] = img[round(size[0]/2)-xMinH:round(size[0]/2)+xMin-xMinH,round(size[1]/2)-yMinH:round(size[1]/2)+yMin-yMinH]
        return toRet

# Scale the image as it would appear if moved from the refDist to the imgDist, extend the image by buffer when we scale before we crop
def scaleImgDist(refDist, imgDist, img, refDim, bufferFactor=1):
    scaleFactor = 1.*refDist**1/imgDist**1
    if scaleFactor > 2000: scaleFactor = 2000.
    #interp = cv2.INTER_LINEAR
    interp = cv2.INTER_NEAREST
    if scaleFactor < 1.0:
        interp = cv2.INTER_AREA
        newImg = img
        #newImg = np.concatenate((img, img, img), axis=0)
        #newImg = np.concatenate((newImg, newImg, newImg), axis=1)
        newImg = cropImg(newImg, tuple([int(round(i/scaleFactor*bufferFactor)) for i in img.shape[:2]]))
    else:
        newImg = cropImg(img, tuple([int(round(i/scaleFactor*bufferFactor)) for i in img.shape[:2]]))
    newImg = cv2.resize(newImg, tuple([int(round(i*scaleFactor)) for i in newImg.shape[:2]]), 0, 0,interp)
    #print newImg.shape
    #print refDim
    newImg = cropImg(newImg, tuple([int(round(i)) for i in refDim]))
    #print newImg.shape
    return newImg

def singleAxisDistort(img, func, axis=1):
    temp = np.swapaxes(img,0,1) if axis==1 else img
    size = temp.shape
    mid = size[0]/2
    toRet = None
    for idx, row in enumerate(temp):
        scale = func(idx-mid)
        if toRet is None:
            toRet = np.zeros((size[0],int(math.ceil(size[1]*scale)),size[2]),dtype=img.dtype)
        scaled = cv2.resize(row, (0,0), fx=1, fy=scale)
        top = (toRet.shape[1]-scaled.shape[0])/2
        bot = top + scaled.shape[0]
        toRet[idx,top:bot] = np.copy(scaled)
    return np.swapaxes(toRet,0,1) if axis==1 else toRet
'''
func = lambda x: 1+.000000092*x**2
simg = singleAxisDistort(img, func)
'''

# Generate the kernel for the PSF of the eye
def getPSF(focalDist, blurDist, aperture=None, pixelDiameter=.00033, **kwargs):
    # should take args for distances to focal plane and blur plane and compute it properly
    if aperture is None:
        aperture = .0035  # Human pupil is between 1.5 and 8 mm
    ''' NOT a gaussian!!!
    ksize = 9
    sigma = 3
    d1Kernel = cv2.getGaussianKernel(ksize, sigma) 
    kernel = d1Kernel * cv2.transpose(d1Kernel)
    '''
    # the kernel is similar to a tophat
    # calculate diameter in pixels
    CoC = getCoC(aperture, focalDist, blurDist)  # this is diameter on the focal plane - but we are convolving, so - should be same on both if we have already scaled them?
    d = CoC/pixelDiameter                        # this is size of pixels at focal plane
    tophat = circle(int(round(d))/2*2+1, int(round(d))/2*2+1, d)
    #if np.sum(tophat) == 0:
    #    tophat[round(d/2),round(d/2)] = [True]
    tophat = tophat[:,~np.all(tophat == 0, axis=0)]   # remove zero columns
    tophat = tophat[~np.all(tophat == 0, axis=1)]     # remove zero rows
    kernel = 1./np.sum(tophat)*tophat
    return kernel
    
# Generate a linear motion blur kernel
def getMotionKernel(ksize, sigma=3):
    d1Kernel = cv2.getGaussianKernel(ksize, sigma) 
    kernel = d1Kernel * np.ones([1,ksize])/ksize
    return kernel

# Generate the image as seen through the mask blurred by the kernel - (mask * PSF) + image
def generateImage(img, mask, kernel):
    if img.dtype.name[:5] != 'float':
        scale = float(getBitDepthScaleFactor(img.dtype.name))
        img = np.float32(img)/scale
    if mask.dtype.name[:5] != 'float':
        scale = float(getBitDepthScaleFactor(mask.dtype.name))
        mask = np.float32(mask)/scale
    if kernel.dtype.name[:5] != 'float':
        scale = float(getBitDepthScaleFactor(kernel.dtype.name))
        kernel = np.float32(kernel)/scale
    dst = cv2.filter2D(mask,-1,kernel)
    dst = cropImg(dst, img.shape)
    final = cv2.multiply(img, dst)
    return final
    
# Render a pre-corrected image so that when viewed through the mask with the given parameters, the original image will be visible
def preCorrectImage(img, mask, focalDist, blurDist, aperture, nsr, maskImg=None):
    psf = getPSF(focalDist,blurDist,aperture)
    deConMask = deconvolveWiener(mask, psf, nsr)
    deConMask = cropImg(deConMask, img.shape)
    preCorImg = deConMask
    preCorImg = cv2.divide(img, deConMask)
    return preCorImg

# Perform a convoultuion of an image by a psf kernel
def convolve(img,psf):
    return cv2.filter2D(img,-1,psf)
    
# Deblur image using Wiener filter
# G(f) = {H^*(f)} / {|H(f)|^2 + 1/NSR}
def deconvolveWiener(img,psf,nsr):
    if img.dtype.name[:5] != 'float':       # convert to float if not already
        scale = float(getBitDepthScaleFactor(img.dtype.name))
        img = np.float32(img)/scale
    if psf.dtype.name[:5] != 'float':       # convert to float if not already
        scale = float(getBitDepthScaleFactor(psf.dtype.name))
        psf = np.float32(psf)/scale
    #img = np.float32(img)/255.
    if len(img.shape) < 3:
        img = img[...,np.newaxis]
    origRes = img.shape
    img = cv2.copyMakeBorder(img, int(round(img.shape[0]/2.)),int(round(img.shape[0]/2.)),int(round(img.shape[1]/2.)),int(round(img.shape[1]/2.)),cv2.BORDER_REFLECT_101)  # extend border
    if len(img.shape) < 3:
        img = img[...,np.newaxis]
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw,0] = psf #np.float32(psf)/255.
    PSF = cv2.dft(psf_pad[:,:,0], flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + 1./nsr)[...,np.newaxis]
    final = np.zeros_like(img)
    for i in range(origRes[2]):  # do each channel seperately
        IMG = cv2.dft(img[:,:,i], flags=cv2.DFT_COMPLEX_OUTPUT)
        RES = cv2.mulSpectrums(IMG, iPSF , 0)
        final[:,:,i] = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    final = np.roll(final, -kh//2, 0)
    final = np.roll(final, -kw//2, 1)
    final = cropImg(final, origRes)
    return final.clip(0.,1.)

if __name__ == '__main__':
    cross = np.array(((0,0,0,0,255,0,0,0,0),        # Create some white crosses for content 
                      (0,0,0,0,255,0,0,0,0),
                      (0,0,0,0,255,0,0,0,0),
                      (0,0,0,0,255,0,0,0,0),
                      (255,255,255,255,255,255,255,255,255),
                      (0,0,0,0,255,0,0,0,0),
                      (0,0,0,0,255,0,0,0,0),
                      (0,0,0,0,255,0,0,0,0),
                      (0,0,0,0,255,0,0,0,0),
                      ),dtype=np.uint8)
    cross = np.dstack((cross,cross,cross))
    spacing = 12
    crosses = np.hstack((cross, np.zeros((cross.shape[0],cross.shape[1]*spacing,3),dtype=np.uint8),cross, np.zeros((cross.shape[0],cross.shape[1]*spacing,3),dtype=np.uint8),cross))
    crosses = np.vstack((crosses,np.zeros((crosses.shape[0]*spacing,crosses.shape[1],3),dtype=np.uint8),crosses,np.zeros((crosses.shape[0]*spacing,crosses.shape[1],3),dtype=np.uint8),crosses))

    cv2.namedWindow('HMD')          # Create the windows
    cv2.namedWindow('Near')
    cv2.namedWindow('Far')
    
    # Displays are an easy way to store the resolution and calculate the pixel size needed for calculating PSF
    display = Display()
    # Manipulate the images
    crosses = cv2.resize(crosses, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) # double size for demo
    near = cropImg(crosses, display.resolution) # will shrink or grow image to resolution
    far = scaleImgDist(.5, 1, crosses, display.resolution, 5) # will scale an image from one distance to another
    hmd = convolve(near, getPSF(.5, 1, aperture=.004, pixelDiameter=sum(display.pixelSize())/2)) # simulated optical blur
    while(True):
        cv2.imshow('HMD', hmd)      # Display the images
        cv2.imshow('Near', near)
        cv2.imshow('Far', far)
        ch = cv2.waitKey() & 0xFF
        if ch == 27:                # escape
            break
    cv2.destroyAllWindows()
	