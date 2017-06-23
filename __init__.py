#!/usr/bin/env python
# pylint: disable=bad-whitespace, line-too-long
'''A library of graphics classes for defining a scene graph by using render stacks

David Dunn
Sept 2014 - Created
Oct 2014 - Added raytracing
July 2015 - Added openGL support
Jan 2017 - Redefined submodules

ALL UNITS ARE IN METERS
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["SceneGraph", "Plug", "WorldObject"]

import sys, operator
import numpy as np
from itertools import chain
if sys.version_info[0] == 2:
    from itertools import imap, izip_longest
else:
    from itertools import zip_longest as izip_longest
    imap = map
import OpenGL.GL as GL

import dGraph.xformMatrix as xm
import dGraph.dio.obj as obj

class SceneGraph(dict):
    ''' A top level object that houses everything in a scene '''
    def __init__(self, name, file=None):
        self._name = name
        self.classifier = 'scene'
        self._children = []
        self.worldMatrix = xm.eye(4)
        self.renderGraphs = set()
        self.cameras = set()
        self.shapes = set()
        self.materials = set()
        self.lights = set()
        # could put the new light and new material members here - and keep track of them here rather than in the class
        # maybe a good place to store a dict of all the objects in the scene? - for searching and stuff  - DONE!!!
    def __hash__(self):
        return hash(self._name)  
    @property
    def name(self):     # a read only attribute
        return self._name
    def add(self, member):
        if member.name in self:
            raise ValueError('Object with name: %s already exists in scene.'%member.name)
        else:
            self[member.name] = member
            # maybe we could categorize them, or create a method for looking for objects of a certain type, ie all lights?
            if member.classifier == 'renderGraph':
                self.renderGraphs.add(member.name)
            elif member.classifier == 'camera':
                self.cameras.add(member.name)
            elif member.classifier == 'shape':
                self.shapes.add(member.name)
            elif member.classifier == 'material':
                self.materials.add(member.name)
            elif member.classifier == 'light':
                self.lights.add(member.name)
        return member
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
    def render(self):
        for rg in self.renderGraphs:
            self[rg].render(0)

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
        self.classifier = 'object'
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
    def category(self):
        return 'object'
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

def initGL():
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_MULTISAMPLE)
    GL.glDepthFunc(GL.GL_GREATER)
    #GL.glDepthFunc(GL.GL_LESS)
    GL.glDepthRange(0,1)
    GL.glClearDepth(0)
    GL.glClearColor(0, 0, 0, 0)

    #GL.glEnable(GL.GL_TEXTURE_2D)                                      # Not needed for shaders?
    #GL.glEnable(GL.GL_NORMALIZE)                                       # Enable normal normalization