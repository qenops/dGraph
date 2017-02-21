#!/usr/bin/env python
'''Implicit shapes submodule for dGraph scene description module

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["ImplicitSurface", "Sphere", "Plane"]

#from dGraph import *
from dGraph.shapes import Shape

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