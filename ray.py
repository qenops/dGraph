#!/usr/bin/env python
'''Ray implementation - for general raytracing

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METERS 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'

import numpy as np
from numpy import dot, cross
from numpy.linalg import norm
import operator
from dGraph.shapes import Shape
    
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
        intersections.sort(key=operator.itemgetter('distance'))
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