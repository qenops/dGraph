#!/usr/bin/env python
'''A library of graphics classes for defining a scene graph

David Dunn
Sept 2014 - Created
Oct 2014 - Added raytracing
July 2015 - Added openGL support

ALL UNITS ARE IN METERS 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.5'

import sys
import numpy as np
from numpy import dot, vdot, cross, matlib
from numpy.linalg import norm
from math import sqrt, sin, cos, pi, floor, ceil
from itertools import chain
if sys.version_info[0] == 2:
    from itertools import imap, izip_longest
else:
    from itertools import zip_longest as izip_longest
    imap = map
import operator 
from operator import itemgetter
from random import uniform
#import OpenGL
#OpenGL.ERROR_CHECKING = False      # Uncomment for 2x speed up
#OpenGL.ERROR_LOGGING = False       # Uncomment for speed up
import OpenGL.GL as GL
import ctypes

import xformMatrix as xm
import objIO as obj
import shaders as dgs

shadows=False
epsilon=0.001
reflectDepth=0
reflectionsMax=1
depth = 0

class SceneGraph(object):
    ''' A top level object that houses everything in a scene '''
    def __init__(self, file=None):
        self._children = []
        self.worldMatrix = matlib.identity(4)
        # could put the new light and new material members here - and keep track of them here rather than in the class
        # maybe a good place to store a dict of all the objects in the scene? - for searching and stuff
    def __iter__(self):
        for v in chain(*imap(iter, self._children)):
            yield v
    def addChild(self, child):
        self._children.append(child)
    def removeChild(self, child):
        self._children.remove(child)
    def getScene(self):
        return self
    def loadScene(self, file):
        if self._children != []:
            #probably some warning about saving existing scene first???
            pass
        pass

class ComparableMixin(object):
    def __eq__(self, other):
        return not self < other and not self > other 
    def __ne__(self, other):
        return self < other or self > other
    def __ge__(self, other):
        return not self < other
    def __le__(self, other):
        return not self > other

class Plug(ComparableMixin):
    ''' A class to define attributes which could contain values or be driven by some other plug's output '''
    def __init__(self, owner, dtype, default=None, accepts='both'):
        self.owner = owner          # who owns this plug?  (needed for dirty propigation)
        self.dtype = dtype          # our data type
        self._accepts = accepts
        self.input = None           # connect to another plug to drive my value
        self.output = []            # connect to other plugs to drive their values (needed for dirty propigation)
        self.function = None        # a function to modify my input plug's value
        default = default if default is not None else self.dtype()
        self.value = default        # store a value
    @property
    def value(self):
        if self.input is None:
            return self._value
        if self.function is None:
            return self.input.value
        return self.function(self.input)
    @value.setter
    def value(self, newValue):
        if self.input is not None:
            raise AttributeError('Cannot set attribute.  It already has an incomming connection')
        if not isinstance(newValue, self.dtype):
            raise TypeError('Cannot set attribute.  %s object is not of type %s.'%(newValue.__class__, self.dtype))
        self._value = newValue
        self.propigateDirty()
    def propigateDirty(self):
        for other in self.output:
            other.owner._matrixDirty = True
            other.owner._parentMatrixDirty()
            other.propigateDirty()
    def connect(self, other):
        ''' connect an input as driver '''
        if not (self._accepts == 'both' or self._accepts == 'in'):
            raise AttributeError('Cannot connect attribute.  Driven does not accept incomming connections.')
        if not (other._accepts == 'both' or other._accepts == 'out'):
            raise AttributeError('Cannot connect attribute.  Driver does not accept outgoing connections.')
        if not issubclass(other.dtype, self.dtype):
            raise TypeError('Cannot connect attribute.  %s input is not of type %s'%(other.dtype, self.dtype))
        self.input = other
        other.output.append(self)
    def connectFunction(self, inputList, func):
        ''' connect an input[s] as driver '''
        if not (self._accepts == 'both' or self._accepts == 'in'):
            raise AttributeError('Cannot connect attribute.  Driven does not accept incomming connections.')
        if not isinstance(inputList, list):
            inputList = [inputList]
        for other in inputList:
            if not (other._accepts == 'both' or other._accepts == 'out'):
                raise AttributeError('Cannot connect attribute.  Driver does not accept outgoing connections.')
        if callable(func):
            self.input = inputList if len(inputList) > 1 else inputList[0]
            self.function = func       # should I verify that it accepts input's dtype and returns my dtype ???
            for other in inputList:
                other.output.append(self)
            return
        raise AttributeError('Cannot set as function.  %s is not callable.'%func)
    def disconnect(self):
        ''' bake the value and disconnect all inputs and functions '''
        self._value = self.value
        try:
            self.input.output.remove(self)
        except AttributeError:
            pass
        self.input = None
        self.function = None
    def __iter__(self):
        return self.value.__iter__()
    def next(self):
        return self.value.next()
    def __getitem__(self, index):
        return self.value[index]
    def __lt__(self, other):
        try:
            return self.value < other.value
        except AttributeError:
            return self.value < other
    def __gt__(self, other):
        try:
            return self.value > other.value
        except AttributeError:
            return self.value > other
    def __repr__(self):
        return self.value.__repr__()
def operate(func):
    def inner(self,other):
        try:
            return func(self.value, other.value)
        except AttributeError:
            return func(self.value, other)
    return inner
def roperate(func):
    def inner(self,other):
        try:
            return func(other.value, self.value)
        except AttributeError:
            return func(other, self.value)
    return inner
for op in ['__add__','__sub__','__mul__','__floordiv__','__div__','__truediv__','__mod__','__pow__','__and__','__or__','__xor__']:
    try:
        setattr(Plug, op, operate(getattr(operator, op)))
        setattr(Plug, '__r%s'%op[2:], roperate(getattr(operator, op)))
    except AttributeError:
        pass

class WorldObject(object):
    ''' Anything that has a transfom in the world 
        Translate
        Rotate
        Scale
        Bounding Box
        Matrix (generates a Transform Matrix for the object)
        Parent
        Children
        Shapes
    '''
    def __init__(self, name, parent):
        self._children = []        
        self._matrix = Plug(self, xm.dtype(),xm.eye(4),'out') 
        self._worldMatrix = Plug(self, xm.dtype(),xm.eye(4), 'out')
        self._matrixDirty = True
        self._worldMatrixDirty = True
        self._parent = Plug(self, object, parent, 'in')
        self.parent = parent
        self._name = name
        self._translate = Plug(self, np.ndarray, np.array([0.,0.,0.]))
        self._rotate = Plug(self, np.ndarray, np.array([0.,0.,0.]))
        self._rotateOrder = Plug(self, tuple, (0,1,2)) # xyz rotate order
        self._scale = Plug(self, np.ndarray, np.array([1.,1.,1.]))
        self._min = Plug(self, np.ndarray, np.array([0.,0.,0.]))
        self._max = Plug(self, np.ndarray, np.array([0.,0.,0.]))
        self._shapes = []
        self.renderable = False
    def __iter__(self):
        yield self
        for v in chain(*imap(iter, self._children)):
            yield v
    def __str__(self):
        return '%s'%self.name
    def __repr__(self):
        return '%s'%self.name
    @property
    def name(self):
        return self._name
    def me(self):
        return self
    @property
    def parent(self):
        return self._parent.value
    @parent.setter
    def parent(self, parent):
        try:
            self._parent.value.removeChild(self)
        except ValueError:
            pass
        self._parent.value = parent
        parent.addChild(self)
        self._parentMatrixDirty()
    def _parentMatrixDirty(self):
        self._worldMatrix.propigateDirty()
        self._worldMatrixDirty = True
        for child in self._children:
            child._parentMatrixDirty()
    @property
    def translate(self):
        return self._translate
    @translate.setter
    def translate(self, value):  # should verify that value has 3 values???
        self._translate.value = np.array(value)
        self._translate.propigateDirty()
        self._matrixDirty = True
        self._parentMatrixDirty()
    @property
    def rotate(self):
        return self._rotate
    @rotate.setter
    def rotate(self, value):  # should verify that value has 3 values???
        self._rotate.value = np.array(value)
        self._rotate.propigateDirty()
        self._matrixDirty = True
        self._parentMatrixDirty()
    @property
    def rotateOrder(self):
        return self._rotateOrder
    @rotateOrder.setter
    def rotateOrder(self, value):  # should verify that value has 3 values???
        self._rotateOrder.value = value
        self._rotateOrder.propigateDirty()
        self._matrixDrity = True
        self._parentMatrixDirty()
    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, value):  # should verify that value has 3 values???
        self._scale.value = np.array(value)
        self._scale.propigateDirty()
        self._matrixDirty = True
        self._parentMatrixDirty()
    @property
    def matrix(self):
        if self._matrixDirty:
            self._matrix.propigateDirty()
            t = xm.calcTranslate(self._translate)
            r = xm.calcRotation(self._rotate, self._rotateOrder)
            s = xm.calcScale(self._scale)
            self._matrix.value = xm.calcTransform(t=t, r=r, s=s)
            self._matrixDirty = False
        return self._matrix
    @property
    def worldMatrix(self):
        if self._worldMatrixDirty:
            self._worldMatrix.value = self.matrix * self.parent.worldMatrix
            self._worldMatrixDirty = False
        return self._worldMatrix
    def localPointToWorld(self, local):
        ''' use the world matrix to calculate a local point in worldspace '''
        return np.array(np.concatenate((local,[1]))*self.worldMatrix.value)[0][0:3]
    def localVectorToWorld(self, local):
        ''' use the world matrix to calculate a local vector in worldspace '''
        return np.array(np.concatenate((local,[0]))*self.worldMatrix.value)[0][0:3]
    def setTranslate(self, tx, ty, tz):
        self.translate = np.array([tx,ty,tz])
    def setRotate(self, rx, ry, rz):
        self.rotate = np.array([rx,ry,rz])
    def setRotateOrder(self, ro):
        self.rotateOrder = ro
    def setScale(self, sx, sy, sz):
        self.scale = np.array([sx,sy,sz])
    def addChild(self, child):
        self._children.append(child)
        #self.boundingBox()
    def removeChild(self, child):
        self._children.remove(child)
        #self.boundingBox()
    def addShape(self, shape):
        self._shapes.append(shape)
        #self.boundingBox()
    def getScene(self):
        if self.parent is None:
            return self
        else:
            return self.parent.getScene()
    def boundingBox(self):
        ''' calculate the local space bounding box of the object '''
        # TODO
        # NO NO NO!!!! This is not respecting local vs world space - need to decide how to deal with that
        #childMin = [child._min for child in self._children]
        #childMax = [child._max for child in self._children]
        #shapeMin = [shape._min for shape in self._shapes]
        #shapeMax = [shape._max for shape in self._shapes]
        #xMin = min(self._translate[0], min([val[0] for val in shapeMin]), min([val[0] for val in childMin]))
        #yMin = min(self._translate[1], min([val[1] for val in shapeMin]), min([val[1] for val in childMin]))
        #zMin = min(self._translate[2], min([val[2] for val in shapeMin]), min([val[2] for val in childMin]))
        #xMax = max(self._translate[0], max([val[0] for val in shapeMax]), max([val[0] for val in childMax]))
        #yMax = max(self._translate[1], max([val[1] for val in shapeMax]), max([val[1] for val in childMax]))
        #zMax = max(self._translate[2], max([val[2] for val in shapeMax]), max([val[2] for val in childMax]))
        #self._min = (xMin, yMin, zMin)
        #self._max = (xMax, yMax, zMax)
        pass

class Shape(WorldObject):
    ''' Any world object that has a renderable surface
        Material (connection to a material class object for illumination information)
        Intersection (given a Ray will return all the intersection points and distances)
        Normal (given a surface point will calculate a normalized normal)
        Bounding Box
        ???? Casts shadows
        ???? U and V values
    '''
    def __init__(self, name, parent):
        super(Shape, self).__init__(name, parent)
        #self.setMaterial(Material.newMaterial('default'))
        self.setMaterial(dgs.Material('default'))
        self.renderable = True
    @property
    def material(self):
        return self._material
    def setMaterial(self, material):
        self._material = material
    def intersection(self, ray):
        ''' an abstract method for implimentation in subclasses'''
        raise NotImplementedError("Please Implement this method")
        return None
    def generateVBO(self):
        ''' an abstract method for implimentation in subclasses'''
        raise NotImplementedError("Please Implement this method")
    def renderGL(self):
        ''' an abstract method for implimentation in subclasses'''
        raise NotImplementedError("Please Implement this method")

class ImplicitSurface(Shape):
    ''' Any surface object whose surface is defined by a mathematical function
        Function
    '''    
    def __init__(self, name, parent, function):
        super(ImplicitSurface, self).__init__(name, parent)
        self._function = function
    @property
    def function(self):
        return self._function
    ''' we need to impliment a intersection method based on function, but don't know how currently '''
    
class Sphere(ImplicitSurface):
    ''' A MathSurface with a spherical math function
        Radius
    '''
    def __init__(self, name, parent, radius=1):
        function = '(x - x_0 )^2 + (y - y_0 )^2 + ( z - z_0 )^2 = r^2'
        super(Sphere, self).__init__(name, function, parent)
        self._radius = radius
    @property
    def radius(self):
        return self._radius
    def setRadius(self, radius):
        self._radius = radius
    def intersection(self, ray):  # equations from wikipedia
        l = ray.vector
        o = ray.point
        c = self.translate
        r = self.radius
        root = ((dot(l,o-c))**2-(dot(o-c,o-c))+r**2)
        dist = []
        if root < 0:    # no intersection
            return None
        if root == 0:   # one intersection
            dist.append(-(dot(l,o-c)))
        else:           # two intersections
            dist.append(-(dot(l,o-c))-sqrt(root))
            dist.append(-(dot(l,o-c))+sqrt(root))
        # Ok so now we have the distances - we need to get the point and normal
        toReturn = []
        for d in dist:
            if d >= 0:
                point = d*l + o
                normal = (point - c)/norm(point - c)
                intersection = {'distance':d,'point':point, 'normal':normal, 'material':self.material}  # will prob want to inclue uv in the future
                toReturn.append(intersection)
        return toReturn
        
class Plane(ImplicitSurface):
    def __init__(self, name, parent, distance=0, normal=[0,1,0]):
        function = '(p-point).dot(normal)=0'
        super(Plane, self).__init__(name, parent, function)
        self._normal = np.array(normal)
        self._distance = distance
    @property
    def normal(self):
        return self._normal
    def setNormal(self, normal):
        self._normal = normal
    @property
    def point(self):
        return self._normal*self._distance
    def setDistance(self, distance):
        self._distance = distance
    def intersection(self, ray):  # equations from wikipedia
        n = self.localVectorToWorld(self.normal)
        p = self.localPointToWorld(self.point)
        d, point = ray.intersectPlane(p,n)
        if d is None:
            return []
        intersection = {'distance':d,'point':point, 'normal':n, 'material':self.material}  # will prob want to inclue uv in the future
        return [intersection]
    
class PolySurface(Shape):
    ''' A surface object defined by polygons
        Verticies - local x,y,z and w of each vert
        UVs - uv position list
        Normals - normal vector list 
        Edges - maybe not needed?
        Faces - dict List consisting of verts, uvs, and normals of 3 or more verts that make up a face
    '''
    def __init__(self, name, parent, file=None, verts=None, uvs=None, normals=None, faces=None, faceUvs=None, faceNormals=None, faceSizes=None, normalize=False):
        super(PolySurface, self).__init__(name, parent)
        if file is not None:
            verts, uvs, normals, faces, faceUvs, faceNormals, faceSizes = obj.load(file, normalize)
        self._verts = np.matrix([]) if verts is None else verts
        self._uvs = np.matrix([]) if uvs is None else uvs
        self._normals = np.matrix([]) if normals is None else normals
        self._faceVerts = np.matrix([]) if faces is None else faces
        self._faceUvs = np.matrix([]) if faceUvs is None else faceUvs
        self._faceNormals = np.matrix([]) if faceNormals is None else faceNormals
        self._faceSizes = [] if faceSizes is None else faceSizes
        self._triWorldMatrix = None
        #self._vertsGL = None
        #self._uvsGL = None
        #self._normsGL = None
        #self._facesGL = None
        self._kdTree = KdTree()
    def writeToObj(self, file):
        return obj.write(file, self._verts, self._uvs, self._normals, self._faceVerts, self._faceUvs, self._faceNormals)
    def getVertList(self):
        ''' returns a list of verts in world space'''
        if self._verts.shape[1] == 3:
            self._verts = np.concatenate((self._verts,np.ones((self._verts.shape[0],1),dtype=np.float32)),axis=1)
        #return (self._verts*self.worldMatrix).getA()
        return self._verts.getA()
    def getNormalList(self):
        ''' returns a list of unit normals in world space'''
        if self._normals.shape[1] == 3:
            self._normals = np.concatenate((self._normals,np.zeros((self._normals.shape[0],1))),axis=1)
        #normals = self._normals*self.worldMatrix
        normals = self._normals
        return (normals/norm(normals)).getA()
    def buildKdTree(self, file=None):
        if file is not None:
            self._kdTree.load(file)
        else:  # TODO construct the kd tree on our own
            pass
    def intersection(self, ray):  
        toReturn = []
        triangles = []
        faces = self.triangulate()
        if self._kdTree == []:  
            triangles = self.triangulate 
        else:  # use kd tree to get list of possible triangle intersections
            triangles = self._kdTree.intersect(ray, 0)
        for tri in [faces[t] for t in triangles]:
            # run through each triangle and determine if it intersects our ray
            verts = tri['verts']
            normal = tri['normal']
            intersects, d, point = ray.intersectTri(*verts, normal=normal)
            if intersects:
                #uvs = tri['uvs']
                intersection = {'distance':d,'point':point,'normal':normal, 'material':tri['material']}  # will prob want to inclue uv in the future
                toReturn.append(intersection)
        return toReturn
    def triangulate(self):
        # triangles should consist of 3 verts in worldspace with corresponding uvs, normals, faceNormal, and a material of that face
        if self.worldMatrix == self._triWorldMatrix:
            return self._triangles
        triangles = []
        verts = self.getVertList()[np.ravel(self._faceVerts),:]
        uvs = [] if self._uvs == [] else self._uvs[np.ravel(self._faceUvs),:]
        norms = [] if self._normals == [] else self.getNormalList()[np.ravel(self._faceNormals),:]
        start=0
        end=0
        for face in self._faceSizes:
            # check to see if there are more than 3 verts on face
            end+=face
            if face == 3:
                myVerts = verts[start:end]
                myUvs = np.zeros((0,4)) if self._uvs == [] else uvs[start:end]
                normals = np.zeros((0,4)) if self._normals == [] else norms[start:end]
                faceNorm = self.calcTriNorm(*list(myVerts[:,:3])+list(normals[:,:3]))  # calculate the normal for the face - should probably calculate the plane and get the cross vector
                triangles.append({'verts':myVerts, 'normals':normals, 'normal':faceNorm, 'material':self._material, 'uvs':myUvs})
            else: # TODO Generate triangles for faces with more than 3 verts
                pass
            start = end
        self._triangles = triangles
        self._triWorldMatrix = self.worldMatrix
        return self._triangles
    def triangulateGL(self):
        ''' Generate openGL triangle lists in VVVVVTTTTTNNNNN form
        Don't need to worry about world matrix - we will do that via model matrix '''
        # TODO something if max faceSizes is greater than 3
        if max(self._faceSizes) == 3:
            # Combine all the positions, normals, and uvs into one array, then remove duplicates - that is our vertex buffer
            maxSize = 2**21                                             # numpy uint64 is 64 bits spread over 3 attributes is 21 bits 2**21/3 is max number of faces
            fuvs = np.zeros_like(self._faceVerts, dtype=np.uint64) if len(self._uvs.A1) < 3 else self._faceUvs.astype(np.uint64)
            fnorms = np.zeros_like(self._faceVerts, dtype=np.uint64) if len(self._normals.A1) < 3 else self._faceNormals.astype(np.uint64)
            f = np.array(self._faceVerts.astype(np.uint64)+(maxSize*fuvs).astype(np.uint64)+((maxSize**2)*fnorms).astype(np.uint64)).ravel()
            fullVerts, faces = np.unique(f, return_inverse=True)        # get the unique indices and the reconstruction(our element array)
            # Build our actual vertex array by getting the positions, normals and uvs from our unique indicies
            vertsGL = self._verts[fullVerts%maxSize].getA1()
            uvsGL = np.zeros((0),dtype=np.float32) if len(self._uvs.A1) < 3 else self._uvs[(fullVerts/maxSize)%maxSize].getA1()
            normsGL = np.zeros((0),dtype=np.float32) if len(self._normals.A1) < 3 else self._normals[fullVerts/(maxSize**2)].getA1()
            return np.concatenate((vertsGL,uvsGL,normsGL)), faces.astype(np.uint32), [len(vertsGL),len(uvsGL),len(normsGL)]
    def generateVBO(self):
        ''' generates OpenGL VBO and VAO objects '''
        #global shader_pos, shader_uvs, shader_norm
        vertsGL, facesGL, lengths = self.triangulateGL()                                                    # make sure our vert list and face list are populated
        self.numTris = len(facesGL)
        if self._material.shader is None:                                                                   # make sure our shader is compiled
            self._material.compileShader()
        self.vertexArray = GL.glGenVertexArrays(1)                                                          # create our vertex array
        GL.glBindVertexArray(self.vertexArray)                                                              # bind our vertex array
        self.vertexBuffer = GL.glGenBuffers(1)                                                              # Generate buffer to hold our vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffer)                                              # Bind our buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertsGL.nbytes, vertsGL, GL.GL_STATIC_DRAW)                     # Send the data over to the buffer
        # Need to figure out stride and normal offset before we start
        stride = 0                                                                                          # stride does not work
        #stride = facesGL[0].nbytes
        #normOffset = self._verts[0].nbytes*lengths[0]+self._verts[0].nbytes*lengths[1]                      # offset will be length of positions + length of uvs
        # Set up the array attributes
        shader_pos = GL.glGetAttribLocation(self._material.shader, 'position')
        shader_uvs = GL.glGetAttribLocation(self._material.shader, 'texCoord')                              # will return -1 if attribute isn't supported in shader
        shader_norm = GL.glGetAttribLocation(self._material.shader, 'normal')                               # will return -1 if attribute isn't supported in shader
        GL.glEnableVertexAttribArray(shader_pos)                                                            # Add a vertex position attribute
        GL.glVertexAttribPointer(shader_pos, 3, GL.GL_FLOAT, False, stride, None)                           # Describe the position data layout in the buffer
        if len(self._uvs.A1) > 2  and shader_uvs != -1:
            GL.glEnableVertexAttribArray(shader_uvs)                                                                        # Add a vertex uv attribute
            GL.glVertexAttribPointer(shader_uvs, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(lengths[0]*facesGL[0].nbytes))   # Describe the uv data layout in the buffer
        if len(self._normals.A1) > 2 and shader_norm != -1:
            GL.glEnableVertexAttribArray(shader_norm)                                                           # Add a vertex uv attribute
            #GL.glVertexAttribPointer(shader_norm, 3, GL.GL_FLOAT, False, stride, None)                         # Describe the uv data layout in the buffer
            GL.glVertexAttribPointer(shader_norm, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p((lengths[0]+lengths[1])*facesGL[0].nbytes))
        # Create face element array
        self.triangleBuffer = GL.glGenBuffers(1)                                                            # Generate buffer to hold our face data
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.triangleBuffer)                                    # Bind our buffer as element array
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, facesGL.nbytes, facesGL, GL.GL_STATIC_DRAW)             # Send the data over to the buffer
        GL.glBindVertexArray( 0 )                                                                           # Unbind the VAO first (Important)
        GL.glDisableVertexAttribArray(shader_pos)                                                           # Disable our vertex attributes
        GL.glDisableVertexAttribArray(shader_uvs) if len(self._uvs.A1) > 2 and shader_uvs != -1 else True
        GL.glDisableVertexAttribArray(shader_norm) if len(self._uvs.A1) > 2 and shader_norm != -1 else True
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)                                                              # Unbind the buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)                                                      # Unbind the buffer
    def renderGL(self, filmMatrix=None, cameraMatrix=None):
        GL.glUseProgram(self._material.shader)
        # multiply the world transform to the camera matrix
        cameraMatrix = matlib.identity(4) if cameraMatrix is None else cameraMatrix
        filmMatrix = matlib.identity(4) if filmMatrix is None else filmMatrix
        modelViewMatrix = np.array(self.worldMatrix*cameraMatrix, dtype=np.float32)
        mvMatrixUniform = GL.glGetUniformLocation(self._material.shader, 'modelViewMatrix')
        GL.glUniformMatrix4fv(mvMatrixUniform, 1, GL.GL_FALSE, modelViewMatrix)
        normalMatrix = np.ascontiguousarray((self.worldMatrix*cameraMatrix).getI().getT()[:3,:3], dtype=np.float32)
        nMatrixUniform = GL.glGetUniformLocation(self._material.shader, 'normalMatrix')
        GL.glUniformMatrix3fv(nMatrixUniform, 1, GL.GL_FALSE, normalMatrix)
        finalMatrix = np.array(modelViewMatrix*filmMatrix, dtype=np.float32)
        fMatrixUniform = GL.glGetUniformLocation(self._material.shader, 'fullMatrix')
        GL.glUniformMatrix4fv(fMatrixUniform, 1, GL.GL_FALSE, finalMatrix)
        # bind our VAO and draw it
        GL.glBindVertexArray(self.vertexArray)
        GL.glDrawElements(GL.GL_TRIANGLES,self.numTris,GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
    @staticmethod
    def calcTriNorm(a,b,c, na=None, nb=None, nc=None):
        normal = cross((b - a),(c - a))
        normal = normal/norm(normal)
        # check to see if it matches sum of vert norms
        if na is not None and nb is not None and nc is not None:
            sum = na+nb+nc
            if dot(normal,sum) < 0:
                normal *= -1
        return normal
    @classmethod
    def polySphere(cls, name, parent, subDivAxis=32, subDivHeight=16):
        numVertices = (subDivHeight - 2) * subDivAxis + 2
        numFaces = (subDivHeight - 2) * (subDivAxis - 1) * 2
        verts = matlib.zeros((0,3),dtype=np.float32)
        for j in xrange(1,subDivHeight-1):
            for i in xrange(subDivAxis):
                theta = float(j)/(subDivHeight-1) * pi
                phi = float(i)/(subDivAxis-1)  * pi * 2
                x = sin(theta) * cos(phi)
                y = cos(theta)
                z = -sin(theta) * sin(phi)
                verts = np.append(verts, np.array([[x,y,z]]), axis=0)
        verts = np.append(verts, np.array([[0,1,0]]), axis=0)
        verts = np.append(verts, np.array([[0,-1,0]]), axis=0)
        # normals = verts  # at least at initialization
        # faces is a list of 3 indicies in verts that make up each face
        faces = matlib.zeros((0,3), dtype=np.uint32)
        faceSizes = []
        for j in xrange(subDivHeight-3):
            for i in xrange(subDivAxis-1):
                faces = np.append(faces, np.array([[j*subDivAxis + i, (j+1)*subDivAxis + (i+1), j*subDivAxis + (i+1)]]), axis=0)
                faces = np.append(faces,np.array([[j*subDivAxis + i, (j+1)*subDivAxis + i, (j+1)*subDivAxis + (i+1)]]), axis=0)
                faceSizes.append(3)
                faceSizes.append(3)
        for i in xrange(subDivAxis-1):
            faces = np.append(faces,np.array([[(subDivHeight-2)*subDivAxis, i, i + 1]]), axis=0)
            faces = np.append(faces,np.array([[(subDivHeight-2)*subDivAxis + 1, (subDivHeight-3)*subDivAxis + (i+1), (subDivHeight-3)*subDivAxis + i]]), axis=0)
            faceSizes.append(3)
            faceSizes.append(3)
        verts = np.matrix(verts,dtype=np.float32)
        faces = np.matrix(faces,dtype=np.uint32)
        return cls(name, parent=parent, verts=verts, normals=verts, faces=faces, faceNormals=faces, faceSizes=faceSizes)
        
class KdTree(list):
    ''' We need to either convert the tree to worldspace or convert the ray to localspace '''
    def __init__(self, file=None):
        if file is not None:
            self.load(file)
        super(KdTree, self).__init__()
    def load(self, file):
        with open(file) as f:
            for line in f:
                node = []
                tokens = line.split()
                if tokens[0] == 'inner{':
                    node.append(0)
                    node.append(map(float, tokens[1:4]))  # min
                    node.append(map(float, tokens[4:7]))  # max
                    node.append(map(int, tokens[8:10]))  # child ids
                    node.append(int(tokens[10]))  # axis x = 0, y = 1, z = 2
                    node.append(float(tokens[11])) # location of plane
                elif tokens[0] == 'leaf{':
                    node.append(1)
                    node.append(map(float, tokens[1:4]))  # min
                    node.append(map(float, tokens[4:7]))  # max
                    node.append(set(map(int, tokens[8:len(tokens)-1])))  # face ids
                self.append(node)
    def intersect(self, ray, node):
        ''' recursively return a list of all faces from the leaf node that intersects the ray '''
        global depth
        #if depth > 3:
        #    return []
        triangles = set()
        bMin = self[node][1]
        bMax = self[node][2]
        # intersect the box
        if ray.intersectBox(bMin,bMax):  # if it intersects:
            if self[node][0] == 1:   # leaf node
                triangles.update(self[node][3])
            else:
                # we could probably use the plane to determine if we should look in left, right or both
                depth += 1
                triangles.update(self.intersect(ray, self[node][3][0])) #left
                triangles.update(self.intersect(ray, self[node][3][1])) #right
                depth -= 1
        return triangles

class Light(WorldObject):
    ''' A world object that casts light 
        Intensity
        Color
    '''
    _lightList = {}                                         # store all existing lights here
    def __new__(cls, name, *args, **kwargs):                # use for creating new materials - keeps track of all existing materials
        if name in cls._lightList.keys():
            if not isinstance(cls._lightList[name],cls):    # do some type checking to prevent mixed results 
                raise TypeError('Light of name "%s" already exists and is type: %s'%(name, type(cls._lightList[name])))
        else:
            cls._lightList[name] = super(Light, cls).__new__(cls, name, *args, **kwargs)
        return cls._lightList[name]
    @classmethod
    def allLights(cls):
        return cls._lightList
    def __init__(self, name, parent, color=(1,1,1), intensity=1, **kwargs):      
        super(Light, self).__init__(name, parent)
        self._color = np.array(color)
        self._intensity = intensity
    @property
    def name(self):
        return self.name
    def illumination(self, distance):
        return self._color*self._intensity
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 position;
};
'''

class PointLight(Light):
    ''' A light with falloff '''
    def __init__(self, name, parent, falloff=0, **kwargs):      
        super(PointLight, self).__init__(name, parent)
        self._falloff = falloff
    def illumination(self, distance):
        ''' need to add calculation for falloff - inverse square law or something equiv '''
        return self._color*self._intensity
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 position;
    float falloff;
};
'''

class DirectionLight(Light):
    ''' A light where position doesn't matter, only a direction vector '''
    def __init__(self, name, parent, direction=(1.,0.,0.), **kwargs):      
        super(PointLight, self).__init__(name, parent)
        self._direction = np.array(direction)
    @property
    def direction(self):
        return self._direction/norm(self._direction)
    def shaderStruct(self):
        return '''
struct Light {
    vec3 color;
    vec3 direction;
};
'''

class StereoCamera(WorldObject):
    ''' A stereo camera class that houses two cameras '''
    def __init__(self, name, parent, switch=False):
        super(StereoCamera, self).__init__(name, parent)
        self.left = Camera('%s_lf'%name,self)
        self.right = Camera('%s_rt'%name, self)
        self._IPD = Plug(self, float,.062)
        self._converganceDepth = Plug(self, float, 1.)
        # We need to zero out our z rotations and instead move them to the left and right cameras
        self._rotateCorrected = Plug(self, np.ndarray, np.array((0.,0.,0.)))
        self._rotate.connectFunction(self.rotate, self.outRotateCorrected)
        self.left.rotate.connectFunction(self.rotate,self.outRotateZ)
        self.right.rotate.connectFunction(self.rotate,self.outRotateZ)
        self._setupDone = True
        self._lfRenderStack = [self.left]
        self._rtRenderStack = [self.right]
        self._switch = switch
        self.left.translate.connectFunction(self.IPD,self.outLeftTranslate)
        self.right.translate.connectFunction(self.IPD,self.outRightTranslate)
    @property
    def IPD(self):
        return self._IPD
    @property
    def converganceDepth(self):
        return self._convergenceDepth
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
        if attr == 'IPD':
            self._IPD.value = value
            #self._IPD.propigateDirty()
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
    def render(self, width, height, renderStack=[], parentFrameBuffer=0, posWidth=0, clear=True):
        #print '%s entering render. %s %s %s'%(self.__class__, self._name, posWidth, clear)
        split = [self.rtRenderStack,self.lfRenderStack] if self._switch else [self.lfRenderStack,self.rtRenderStack] # switch for crosseye renders
        for idx, stack in enumerate(split):                          # Do left stack then right stack
            #print stack
            temp=stack.pop()
            temp.render(width/2, height, stack, parentFrameBuffer, posWidth=idx*width/2, clear=not idx)   # Go up the render stack to get our texture
        #print '%s leaving render. %s'%(self.__class__, self._name)
    
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
    def setFOV(self, fov):
        self._film.fov = fov
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
            for i in xrange(self._samples):
                ''' random or divided or divided random '''  # this is random distribution
                x = uniform(pixel[0][0],pixel[1][0])
                y = uniform(pixel[0][1],pixel[1][1])
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
    def render(self, width, height, renderStack=[], parentFrameBuffer=0, posWidth=0, clear=True):
        #print '%s entering render. %s %s %s'%(self.__class__, self._name, posWidth, clear)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, parentFrameBuffer)          # Render to our parentFrameBuffer, not screen
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
    def __init__(self, name, parent, worldDim=(.2, .2), res=(512,512), samples=1, near=.1, far=1000., fov=50.):
        if parent is None or not isinstance(parent, Camera):
            raise AttributeError('Parent of a FilmBack must be a Camera.')
        super(FilmBack, self).__init__(name, parent)
        self._worldDim = worldDim
        self.setResolution(res)
        self._near = near
        self._far = far
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
                        for y in xrange(yMin, yMax+1):
                            for x in xrange(xMin, xMax+1):
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
    
class Ray(object):
    ''' A ray shot from a point in a particular direction for calculating intersections in generating shadows and general rendering
        point
        direction
        evaluate (intersect ray with all objects in scene and evaluate t for each)
        possible tMin, tMax
    '''
    def __init__(self, point, vector):
        self._pnt = point
        self._vec = vector/norm(vector)
        self._invec = 1.0/self._vec
    @property
    def point(self):
        return self._pnt
    @property
    def vector(self):
        return self._vec
    def evaluate(self, scene):
        ''' intersect ray with all children objects of scene and return list of intersections
        (walk the graph using bouding boxes on the transforms - 
            if it intersects:
                evaluate children
                if it is a surface:
                    if it intersects:
                        get point and t value (from Surface)
                        may intersect multiple times - need to do something about that
                    else return None)
        That is pie in the sky, for now just brute force it, call the intersect method on every surface object in scene
        '''
        intersections = []
        for obj in scene:
            if isinstance(obj, Shape):
                possibles = obj.intersection(self)  # distance, point, normal, material
                if possibles is not None:
                    for p in possibles:
                        p['object'] = obj
                        intersections.append(p)
        return intersections
    def render(self, world):
        # evaluate the collisions the ray and find the closest
        intersections = self.evaluate(world)
        intersections.sort(key=itemgetter('distance'))
        # calculate color
        if intersections == []:
            return np.array([0,0,0])
        else:
            first = intersections[0]
            return first['material'].render(first['point'], first['normal'], viewVector=self.vector, world=world)
    @classmethod
    def _calcVector(cls, basePoint, goalPoint):
        ''' calculate the vector that starts at basePoint and travels through goalPoint '''
        return goalPoint - basePoint
    def intersectBox(self, bMin, bMax):
        tx1 = (bMin[0]-self._pnt[0])*self._invec[0]
        tx2 = (bMax[0]-self._pnt[0])*self._invec[0]
        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)
        ty1 = (bMin[1]-self._pnt[1])*self._invec[1]
        ty2 = (bMax[1]-self._pnt[1])*self._invec[1]
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
        tz1 = (bMin[2]-self._pnt[2])*self._invec[2]
        tz2 = (bMax[2]-self._pnt[2])*self._invec[2]
        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))
        return tmax >= max(0.0, tmin)
    def intersectBoxOld(self, bMin, bMax):
        cMin = [-float('inf'),-float('inf'),-float('inf')]
        cMax = [float('inf'),float('inf'),float('inf')]
        if self._vec[0] != 0:
            cMin = self._vec*((bMin[0]-self._pnt[0])/self._vec[0]) + self._pnt  # get x min
            cMax = self._vec*((bMax[0]-self._pnt[0])/self._vec[0]) + self._pnt   # get x max
            if (cMin[0]-self._pnt[0])/self._vec[0] < 0 and (cMax[0]-self._pnt[0])/self._vec[0] < 0:  # verify it crosses ray after start point
                return False #, cMin, cMax
        if self._vec[1] != 0:
            yMin = self._vec*((bMin[1]-self._pnt[1])/self._vec[1]) + self._pnt # get y min
            yMax = self._vec*((bMax[1]-self._pnt[1])/self._vec[1]) + self._pnt  # get y max
            if (yMin>cMax).all() or (cMin>yMax).all():
                return False #, cMin, cMax
            if (cMin<yMin).all():
                cMin = yMin
            if (cMax>yMax).all():
                cMax = yMax
            if (cMin[1]-self._pnt[1])/self._vec[1] < 0 and (cMax[1]-self._pnt[1])/self._vec[1] < 0:  # verify it crosses ray after start point
                return False #, cMin, cMax
        if self._vec[2] != 0:
            zMin = self._vec*((bMin[2]-self._pnt[2])/self._vec[2]) + self._pnt # get z min
            zMax = self._vec*((bMax[2]-self._pnt[2])/self._vec[2]) + self._pnt  # get z max
            if (zMin>cMax).all() or (cMin>zMax).all():
                return False #, cMin, cMax
            if (cMin<zMin).all():
                cMin = zMin
            if (cMax>zMax).all():
                cMax = zMax
            if (cMin[2]-self._pnt[2])/self._vec[2] < 0 and (cMax[2]-self._pnt[2])/self._vec[2] < 0:  # verify it crosses ray after start point
                return False #, cMin, cMax
        #if (cMin<self._pnt).all():
        #    cMin = self._pnt
        #if (cMax>self._pnt).all():
        #    cMax = self._pnt
        return True #, cMin, cMax
    def intersectPlane(self, point, normal):
        denom = dot(self._vec,normal)
        if denom == 0:
            return None, None
        dist = (dot(point-self._pnt,normal))/denom
        if dist <= 0:
            return None, None
        point = dist*self._vec + self._pnt
        return dist, point
    def intersectTriSlow(self, a, b, c, normal=None):  # Moller-Trumbore algorithm
        global epsilon
        # Find vectors for two edges sharing point 'a'
        e1 = b - a
        e2 = c - a
        # Begin calculating determinant - also used to calculate 'u' parameter
        P = cross(self._vec, e2)
        # if determinant is near zero, ray lies in plane of triangle
        det = dot(e1, P)
        # NOT CULLING
        if det > -epsilon and det < epsilon:
            return False, None, None
        inv_det = 1.0 / det
        # calculate distance from 'a' to ray origin
        T = self._pnt - a
        # Calculate u parameter and test bound
        u = dot(T, P) * inv_det
        # The intersection lies outside of the triangle
        if u < 0 or u > 1:
            return False, None, None
        # Prepare to test 'v' parameter
        Q = cross(T, e1)
        # Calculate 'v' parameter and test bound
        v = dot(self._vec, Q) * inv_det
        # The intersection lies outside of the triangle
        if v < 0 or u+v > 1:
            return False, None, None
        t = dot(e2, Q) * inv_det
        if(t > epsilon): #ray intersection
            dist, point = self.intersectPlane(a, normal)
            return True, dist, point
        # No intersection
        return False, None, None
    def intersectTri(self, a, b, c, normal=None):
        if normal is None:
            normal = PolySurface.calcTriNorm(a,b,c)
        dist, point = self.intersectPlane(a, normal)
        if dist is None:
            return False, None, None
        # calculate baycentric
        v0 = c - a
        v1 = b - a
        v2 = point - a
        # get dot products
        dot00 = dot(v0, v0)
        dot01 = dot(v0, v1)
        dot02 = dot(v0, v2)
        dot11 = dot(v1, v1)
        dot12 = dot(v1, v2)
        # get barycentric coordinates
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        inside = u>=0 and v>=0 and u+v<1
        return inside, dist, point
    def intersectRay(self, other):
        # get the distance and midpoint of the line of smallest distance between two lines
        v1 = self.vector
        v2 = other.vector
        p1 = self.point
        p2 = other.point
        X = np.cross(v1,v2)
        ray1Pnt = p1 + np.dot(np.cross(p2-p1,v2),X)/np.dot(X,X)*v1
        ray2Pnt = p2 + np.dot(np.cross(p2-p1,v1),X)/np.dot(X,X)*v2
        midPnt = (ray1Pnt+ray2Pnt)/2
        distance = norm(ray2Pnt-ray1Pnt)
        return (distance, midPnt, ray1Pnt, ray2Pnt)
    def projectPointOnRay(self, pnt):
        diff = pnt - self.point 
        return np.dot(diff,self.vector)*self.vector+self.point
    def distanceToPoint(self, pnt):
        diff = self.point - pnt
        return norm(diff-(np.dot(diff,self.vector)*self.vector))
