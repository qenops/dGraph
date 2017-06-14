#!/usr/bin/env python
'''Camera submodule for dGraph scene description module

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Camera", "StereoCamera"]

from dGraph import *
import dGraph.xformMatrix as xm
import dGraph.ui as ui
import OpenGL.GL as GL
import random
from math import floor, ceil
import numpy as np
from numpy.linalg import norm
from numpy import matlib

class Camera(WorldObject):
    ''' A world object from which we can render - renders in -z direction of camera
        Focal Length
        Aperture
        Shutter Speed
        FilmBack (connection to a imagePlane object for rendering)
    '''
    def __init__(self, name, parent):
        super(Camera, self).__init__(name, parent)
        self._film = FilmBack(name='%s_film'%name, parent=self, worldDim=(.2, .2), res=(512,512))
        self._film.setTranslate(0,0,-.1)
        self._samples = 1
        self.renderPixel = self._film.renderPixel
        self._cameraMatrix = Plug(self, xm.dtype(),xm.eye(4), 'out')
        self._cameraMatrixDirty = True
        #focalLength
        #aperture
        #shutterSpeed
    def _parentMatrixDirty(self):
        self._worldMatrixDirty = True
        self._cameraMatrixDirty = True
        for child in self._children:
            child._parentMatrixDirty()
    def setResolution(self, res):
        self._film.setResolution(res)
    @property
    def fov(self):
        return self._film.fov
    @fov.setter
    def fov(self, value):
        self._film.fov = value
    @property
    def near(self):
        return self._film.near
    @near.setter
    def near(self, value):
        self._film.near = value
    @property
    def far(self):
        return self._film.far
    @far.setter
    def far(self, value):
        self._film.far = value
    def setFOV(self, fov):
        self.fov = fov
    def setSamples(self, samples):
        self._samples = samples
    def setBackFaceCulling(self, bfc):
        self._film.setBackFaceCulling(bfc)
    @property
    def cameraMatrix(self):
        if self._cameraMatrixDirty:
            self._cameraMatrix.value = self.worldMatrix*self._film.filmMatrix
        return self._cameraMatrix
    @property
    def filmMatrix(self):
        return self._film.filmMatrix
    def getRays(self, pixel):
        rayList = []
        samplePoints = []
        if self._samples == 1:  # special case - just take the midpoint
            samplePoints.append((pixel[0]+pixel[1])/2)
        else:
            for i in range(self._samples):
                ''' random or divided or divided random '''  # this is random distribution
                x = random.uniform(pixel[0][0],pixel[1][0])
                y = random.uniform(pixel[0][1],pixel[1][1])
                samplePoints.append(np.array([x,y,pixel[0][2]]))
        focalPoint = self.localPointToWorld(np.array([0, 0, 0]))
        for point in samplePoints:
            vector = Ray._calcVector(focalPoint,point)
            if pixel[2]:        # if pic is inverted - (film back is behind focal point)
                vector *= -1
            rayList.append(Ray(focalPoint, vector))
        return rayList
    def raster(self, mode):
        return self._film.raster(self.getScene(), mode)
    def setup(self, width, height):
        ''' just an empty method for compatability with the render stack '''
        return set(self.getScene())
    def render(self, width, height, renderStack=[], parentTextures=[], parentFrameBuffers=[], posWidth=0, clear=True):
        #print '%s entering render. %s %s %s'%(self.__class__, self._name, posWidth, clear)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, (parentFrameBuffers[0] if len(parentFrameBuffers) > 0 else 0))          # Render to our parentFrameBuffer, not screen
        if clear:
            #print '%s clearing. %s'%(self.__class__, self._name)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # should check if renderStack != [] and error or something
        sceneGraph = self.getScene()
        GL.glViewport(posWidth, 0, width, height)                      # set the viewport to the portion we are drawing
        cameraMatrix = self.worldMatrix                               # get the camera matrix
        filmMatrix = self.filmMatrix
        # get the lights in the scene
        for obj in sceneGraph:                                          # Draw the renderable objects in the scene
            if obj.renderable:
                obj.renderGL(filmMatrix,cameraMatrix)
        #print '%s leaving render. %s'%(self.__class__, self._name)
        
class FilmBack(WorldObject):
    ''' A plane connected to a camera for rendering or displaying an image
        WorldDim (w,h) (should we do this diagonally - ie 35 mm?)
        Resolution (w,h)
        Samples (per pixel, for anti-aliasing)
        iter over pixels returning center point for next
            nextPixel
            nextSample (of same pixel)
    '''
    def __init__(self, name, parent, worldDim=(.2, .2), res=(512,512), samples=1, near=1, far=1000., fov=50.):
        if parent is None or not isinstance(parent, Camera):
            raise AttributeError('Parent of a FilmBack must be a Camera.')
        super(FilmBack, self).__init__(name, parent)
        self._worldDim = worldDim
        self.setResolution(res)
        self.near = near
        self.far = far
        self.fov = fov
        self._backFaceCull = True
        self._filmMatrix = None
    @property
    def fov(self):
        return self._fov
    @fov.setter
    def fov(self, value):
        self._fov = value
        self._filmMatrixDirty()
    @property
    def near(self):
        return self._near
    @near.setter
    def near(self, value):
        self._near = value
        self._filmMatrixDirty()
    @property
    def far(self):
        return self._far
    @far.setter
    def far(self, value):
        self._far = value
        self._filmMatrixDirty()
    def setResolution(self, res):
        self._resolution = res
        self._filmMatrixDirty()
    def setBackFaceCulling(self, bfc):
        self.backFaceCull = bfc
    def _filmMatrixDirty(self):
        self._filmMatrix = None
    @property
    def filmMatrix(self):
        if self._filmMatrix is None:
            n = -self._near
            f = -self._far
            r = n*(self._worldDim[0]/2.0)/self.translate[2]  # r = nx/t - similar triangles, but what if we shift, or worse tilt, the film back?
            l = n*(-self._worldDim[0]/2.0)/self.translate[2]
            t = n*(self._worldDim[1]/2.0)/self.translate[2]
            b = n*(-self._worldDim[1]/2.0)/self.translate[2]
            nx = self._resolution[0]
            ny = self._resolution[1]
            '''persp = np.matrix([[n,    0,  0,      0]
                                ,[0,    n,  0,      0]
                                ,[0,    0,  n+f,    1]
                                ,[0,    0,  -f*n,   0]]) 
            proj = np.matrix([[2.0/(r-l),     0,              0,                 0],
                                [0,             2.0/(t-b),      0,                 0],
                                [0,             0,              2.0/(n-f),         0],
                                [-(r+l)/(r-l),  -(t+b)/(t-b),   -(n+f)/(n-f),      1]])
            pixel = np.matrix([[nx/2.0,       0,          0,      0]
                                ,[0,            ny/2.0,     0,      0]
                                ,[0,            0,          1,      0]
                                ,[(nx-1)/2.0,   (ny-1)/2.0, 0,      1]])
            '''
            #persp = xm.calcPerspective(n,f)
            #proj = xm.calcProjection(r,l,t,b,n,f)
            #pixel = xm.calcScreen(nx, ny)
            persp = xm.calcPerspGL(self.fov,float(nx)/ny,n,f)
            proj = xm.calcOrthoGL(r,l,t,b,n,f)
            pixel = matlib.identity(4)
            self._filmMatrix = persp*proj*pixel
        return self._filmMatrix
    def raster(self, scene, mode):
        ''' render via rasterization for all objects under worldObject 
        shading mode:
            0 = wireframe (not currently supported)
            1 = flat shaded
            2 = per-vertex shading
            3 = smooth shaded
        '''
        # note: we are going to use the lignts in the lightList from class Light, rather than lights under our worldObject #
        frameBuffer = np.zeros((self._resolution[1], self._resolution[0], 3))
        depthBuffer = np.ones((self._resolution[1], self._resolution[0]))
        nx = self._resolution[0]
        ny = self._resolution[1]
        for obj in scene:
            if isinstance(obj, Shape):
                # triangles should consist of 3 verts in worldspace with corresponding normals, uvs and a material
                triangles = obj.triangulate()  # objects should know how to transform themselves to worldspace triangles
                i=0
                for tri in triangles:
                    i += 1
                    # transform to camera space - camera space != world space
                    # read face normal - if negative and culling is on, then skip
                    if self._backFaceCull and tri['normal'][2] < 0:
                        continue
                    # transform to image space
                    vertCoord = []
                    pixelCoord = []
                    for vert in tri['verts']:
                        #coord = vert*persp*proj*pixel
                        coord = vert*obj.worldMatrix*self.parent.worldMatrix.getI()*self.filmMatrix*xm.calcScreen(nx, ny)
                        if abs(coord[0,3]) <= .001:
                            print('vert: %s  - Homogenous Coord == 0'%vert)
                        coord = coord/coord[0,3]  # normalize homogenous coordinate
                        pixelCoord.append([coord[0,0],coord[0,1],coord[0,2]])
                        vertCoord.append(np.array(vert*obj.worldMatrix)[:-1])
                    # figure out what pixels it covers:
                    A = pixelCoord[0]
                    B = pixelCoord[1]
                    C = pixelCoord[2]
                    if mode >= 0:  # wireFrame mode
                        # just use midpoint algorithm to draw the 3 edges
                        #pass
                    #else:
                        #verify if in view frustum - cull if not
                        xMin = floor(min(A[0],B[0],C[0]))
                        xMax = ceil(max(A[0],B[0],C[0])) 
                        yMin = floor(min(A[1],B[1],C[1]))
                        yMax = ceil(max(A[1],B[1],C[1]))
                        zMin = floor(min(A[2],B[2],C[2]))
                        zMax = ceil(max(A[2],B[2],C[2]))
                        # I would like to do near/far based on if they are <-1 and >1 but when they cross camera it jumps from pos inf to neg inf
                        if xMax < 0 or xMin > self._resolution[0] or yMax < 0 or yMin > self._resolution[1]:
                            continue;  #next triangle - not in view frustum
                        xMin = max(0,int(xMin))
                        xMax = min(self._resolution[0]-1,int(xMax))
                        yMin = max(0,int(yMin))
                        yMax = min(self._resolution[1]-1,int(yMax))
                        calcBeta = lambda x,y:((A[1]-C[1])*x+(C[0]-A[0])*y+(A[0]*C[1])-(C[0]*A[1]))/((A[1]-C[1])*B[0]+(C[0]-A[0])*B[1]+(A[0]*C[1])-(C[0]*A[1]))
                        calcGamma = lambda x,y:((A[1]-B[1])*x+(B[0]-A[0])*y+(A[0]*B[1])-(B[0]*A[1]))/((A[1]-B[1])*C[0]+(B[0]-A[0])*C[1]+(A[0]*B[1])-(B[0]*A[1]))
                        beta = calcBeta(xMin,yMin)
                        gamm = calcGamma(xMin,yMin)
                        Bx = calcBeta(xMin+1, yMin)-calcBeta(xMin, yMin)
                        Gx = calcGamma(xMin+1,yMin)-calcGamma(xMin,yMin)
                        By = calcBeta(xMin, yMin+1)-calcBeta(xMin, yMin)
                        Gy = calcGamma(xMin,yMin+1)-calcGamma(xMin,yMin)
                        dist = (xMax - xMin) + 1
                        # do some preloop shading calculations
                        color = np.array([1,1,1])
                        colA = np.array([1,1,1])
                        colB = np.array([1,1,1])
                        colC = np.array([1,1,1])
                        if mode == 1:  # flat shaded mode - calculate shading for whole triangle
                            centroid = sum(vertCoord)/3
                            view = centroid/norm(centroid)
                            color = tri['material'].render(point=centroid, normal=tri['normal'], viewVector=view)
                        if mode == 2:  # vert shaded mode - calculate shading for each vert
                            view = vertCoord[0]/norm(vertCoord[0])
                            colA = tri['material'].render(point=vertCoord[0], normal=tri['normals'][0], viewVector=view)
                            view = vertCoord[1]/norm(vertCoord[1])
                            colB = tri['material'].render(point=vertCoord[1], normal=tri['normals'][1], viewVector=view)
                            view = vertCoord[2]/norm(vertCoord[2])
                            colC = tri['material'].render(point=vertCoord[2], normal=tri['normals'][2], viewVector=view)
                        for y in range(yMin, yMax+1):
                            for x in range(xMin, xMax+1):
                                if beta > 0 and gamm > 0 and (beta + gamm) < 1:
                                    #A(x,y) = A[a]+beta*(B[a]-A[a])+gamm*(C[a]-A[a])
                                    # calc depth - use the image z coord
                                    depth = (1/A[2])+beta*((1/B[2])-(1/A[2]))+gamm*((1/C[2])-(1/A[2]))
                                    depth = 1/depth
                                    # if depth is greater than the current pixel buffer  - skip
                                    if depth > depthBuffer[x][y]:
                                        continue
                                    # calculate color for pixel and store
                                    if mode == 2:  # vertex shaded mode - interp between 3 vert colors
                                        color = colA+beta*(colB-colA)+gamm*(colC-colA)
                                    if mode == 3:  # pixel shaded mode - interp between 3 vert positions and normals
                                        point = vertCoord[0]+beta*(vertCoord[1]-vertCoord[0])+gamm*(vertCoord[2]-vertCoord[0])
                                        view = point/norm(point)
                                        normal = tri['normals'][0]+beta*(tri['normals'][1]-tri['normals'][0])+gamm*(tri['normals'][2]-tri['normals'][0])
                                        color = tri['material'].render(point=point, normal=normal, viewVector=view)
                                    frameBuffer[ny-y,x] = color # [(depth+2)/2,(depth+2)/2,(depth+2)/2]
                                    depthBuffer[ny-y,x] = depth
                                beta += Bx
                                gamm += Gx
                            beta += By - dist*Bx
                            gamm += Gy - dist*Gx
        return frameBuffer, depthBuffer
    def renderPixel(self, col, row):
        ''' use rayTracing to calculate color of a pixel '''
        pixel = self.getPixelPos(col, row)
        rayList = self._parent.getRays(pixel)
        world = self.getScene()
        colors = []
        for ray in rayList:
            colors.append(ray.render(world))
        #get the mean
        mean = sum(colors)/float(len(colors))
        # return the mean color
        return mean
    def getPixelPos(self, col, row):
        xRes = self._resolution[0]
        yRes = self._resolution[1]
        width = self._worldDim[0]
        height = self._worldDim[1]
        invert = False
        #row = yRes - row                       # if displayed image starts from top
        if self._translate[2] > 0:              # if filmBack is behind focalpoint, invert image
            row = yRes - row
            col = xRes - col
            invert = True
        left = col*(float(width)/xRes)-.5*width
        right = (col+1)*(float(width)/xRes)-.5*width
        top = row*(float(height)/yRes)-.5*height
        bot = (row+1)*(float(height)/yRes)-.5*height
        leftTop = self.localPointToWorld([left, top, 0.0])
        rightBot = self.localPointToWorld(np.array([right, bot, 0.0]))
        return (leftTop, rightBot, invert)

class StereoCamera(WorldObject):
    ''' A stereo camera class that houses two cameras '''
    def __init__(self, name, parent, switch=False):
        super(StereoCamera, self).__init__(name, parent)
        self.left = Camera('%s_lf'%name,self)
        self.right = Camera('%s_rt'%name, self)
        self._ipd = Plug(self, float,.062)
        self._converganceDepth = Plug(self, float, 1.)
        self._midOffset = Plug(self, int,0)
        # We need to zero out our z rotations and instead move them to the left and right cameras
        self._rotateCorrected = Plug(self, np.ndarray, np.array((0.,0.,0.)))
        self._rotate.connectFunction(self.rotate, self.outRotateCorrected)
        self.left.rotate.connectFunction(self.rotate,self.outRotateZ)
        self.right.rotate.connectFunction(self.rotate,self.outRotateZ)
        self._setupDone = True
        self._lfRenderStack = ui.RenderStack([self.left])
        self._rtRenderStack = ui.RenderStack([self.right])
        self._switch = switch
        self.left.translate.connectFunction(self.ipd,self.outLeftTranslate)
        self.right.translate.connectFunction(self.ipd,self.outRightTranslate)
    @property
    def ipd(self):
        return self._ipd
    @property
    def converganceDepth(self):
        return self._convergenceDepth
    @property
    def midOffset(self):
        return self._midOffset
    @midOffset.setter
    def midOffset(self, value):
        self._midOffset = value
    @staticmethod
    def outLeftTranslate(input):
        return np.array((input/2.,0.,0.))
    @staticmethod
    def outRightTranslate(input):
        return np.array((input/-2.,0.,0.))
    @property
    def rotate(self):
        ''' We need to zero out our z rotations and instead move them to the left and right cameras '''
        return self._rotateCorrected
    @rotate.setter
    def rotate(self, value):  # should verify that value has 3 values???
        self._rotateCorrected.value = np.array(value)
        self._rotateCorrected.propigateDirty()
        self._matrixDirty = True
        self._parentMatrixDirty()
    @staticmethod
    def outRotateCorrected(rotation):
        return np.array((rotation[0],rotation[1],0.))
    @staticmethod
    def outRotateZ(rotation):
        return np.array((0.,0.,rotation[2]))
    @property
    def lfRenderStack(self):
        return list(self._lfRenderStack)
    @property
    def rtRenderStack(self):
        return list(self._rtRenderStack)
    def leftStackAppend(self, value):
        self._lfRenderStack.append(value)
    def rightStackAppend(self, value):
        self._rtRenderStack.append(value)
    def stackAppend(self, value):
        self.leftStackAppend(value)
        self.rightStackAppend(value)
    def stackSuspend(self):
        self._lfRenderStackHold = self._lfRenderStack
        self._rtRenderStackHold = self._rtRenderStack
        self._lfRenderStack = [self.left]
        self._rtRenderStack = [self.right]
    def stackResume(self):
        self._lfRenderStack = self._lfRenderStackHold
        self._rtRenderStack = self._rtRenderStackHold
    def __setattr__(self, attr, value):
        #print 'setting %s to %s'%(attr,value)
        if attr == 'ipd':
            self._ipd.value = value
            #self._ipd.propigateDirty()
        elif attr == 'converganceDepth':
            self._convergenceDepth.value = value
            #self._converganceDepth.propigateDirty()
            # change the cameras in some manner (probably translating the film back)
        elif '_setupDone' not in self.__dict__ or attr in dir(self):            # If we are still setting up or attribute exists
            super(StereoCamera, self).__setattr__(attr, value)
        else:                                                                   # Take a look at my cameras if attr doesn't exist
            setattr(self.left,attr,value)
            setattr(self.right,attr,value)
    def __getattr__(self, attr):                                                # If attr doesn't belong to me assume it is in my camera
        if hasattr(self.left, attr):
            if callable(getattr(self.left, attr)):
                def wrapper(*args, **kw):
                    #print('called with %r and %r' % (args, kw))
                    lf = getattr(self.left, attr)(*args, **kw)
                    rt = getattr(self.right, attr)(*args, **kw)
                    truth = lf == rt
                    if isinstance(truth, np.ndarray):
                        truth = np.all(lf==rt)
                    if truth:
                        return lf
                    return (lf,rt)
                return wrapper
            lf = getattr(self.left, attr)
            rt = getattr(self.right, attr)
            truth = lf == rt
            if isinstance(truth, np.ndarray):
                truth = np.all(lf==rt)
            if truth:
                return lf
            return (lf,rt)
        raise AttributeError(attr)
    def setup(self, width, height):
        sceneGraphSet = set(self.getScene())
        for node in set(self.lfRenderStack+self.rtRenderStack):
            sceneGraphSet.update(node.setup(width/2, height))
        return sceneGraphSet
    def render(self, width, height, renderStack=[], parentTextures=[], parentFrameBuffers=[], posWidth=0, clear=True):
        #print('%s entering render. %s %s %s'%(self.__class__, self._name, posWidth, clear))
        split = [self.rtRenderStack,self.lfRenderStack] if self._switch else [self.lfRenderStack,self.rtRenderStack] # switch for crosseye renders
        newWidth = (width - self._midOffset)/2
        for idx, stack in enumerate(split):                          # Do left stack then right stack
            #print(stack)
            temp=stack.pop()
            temp.render(int(width/2), height, stack, textures, parentFrameBuffers, posWidth=int((idx*newWidth)+((idx*2-1)*self._midOffset)), clear=not idx)   # Go up the render stack to get our texture
        #print('%s leaving render. %s'%(self.__class__, self._name))