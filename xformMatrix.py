#!/usr/bin/env python
'''Transform matrix methods compatable with Maya's transform

A set of 3d transformation matrix methods developed for use in 3d graphics applications
impliments many functions for easier use in composing and decomposing transform matrices.

***************************************************************************************************************************
******** NOTE:  This matrix is constructed with homogenous coordinates on the right and translations on the bottom ********
********                                  And are therefore POST-multiplied                                        ********
***************************************************************************************************************************

David Dunn
July 2011 - created
July 2015 - removed class and made more import friendly
www.qenops.com

---------------------------------------------------------------------------------------------------------------------------
FROM THE MAYA DOCS:
A transformation matrix is composed of the following components, all components with units will be in maya's internal units (radians for rotations and centimeters for translations):

Scale pivot point           [Sp]    point around which scales are performed 
Scale                       [S]     scaling about x, y, z axes 
Shear                       [Sh]    shearing in xy, xz, yx 
Scale pivot translation     [St]    translation introduced to preserve existing scale transformations when moving pivot. This is used to prevent the object from moving when the objects pivot point is not at the origin and a non-unit scale is applied to the object.
Rotate pivot point          [Rp]    point about which rotations are performed 
Rotation orientation        [Ro]    rotation to orient local rotation space 
Rotation                    [R]     rotation 
Rotate pivot translation    [Rt]    translation introduced to preserve exisitng rotate transformations when moving pivot. This is used to prevent the object from moving when the objects pivot point is not at the origin and the pivot is moved. 
Translate                   [T]     translation in x, y, z axes 

The matrices are post-multiplied in Maya. For example, to transform a point P from object-space to world-space (P') you would need to post-multiply by the worldMatrix. (P' = P x WM)

The transformation matrix is then constructed as follows:

     -1                       -1
  [Sp]x[S]x[Sh]x[Sp]x[St]x[Rp]x[Ro]x[R]x[Rp]x[Rt]x[T]
where 'x' denotes matrix multiplication and '-1' denotes matrix inversion

     Sp = |  1    0    0    0 |     St = |  1    0    0    0 |
          |  0    1    0    0 |          |  0    1    0    0 |
          |  0    0    1    0 |          |  0    0    1    0 |
          | spx  spy  spz   1 |          | sptx spty sptz  1 |

     S  = |  sx   0    0    0 |     Sh = |  1    0    0    0 |
          |  0    sy   0    0 |          | shxy  1    0    0 |
          |  0    0    sz   0 |          | shxz shyz  1    0 |
          |  0    0    0    1 |          |  0    0    0    1 |

     Rp = |  1    0    0    0 |     Rt = |  1    0    0    0 |
          |  0    1    0    0 |          |  0    1    0    0 |
          |  0    0    1    0 |          |  0    0    1    0 |
          | rpx  rpy  rpz   1 |          | rptx rpty rptz  1 |

     Ro = AX * AY * AZ

     AX = |  1    0    0    0 |     AY = |  cy   0   -sy   0 |
          |  0    cx   sx   0 |          |  0    1    0    0 |
          |  0   -sx   cx   0 |          |  sy   0    cy   0 |
          |  0    0    0    1 |          |  0    0    0    1 |

     AZ = |  cz   sz   0    0 |     sx = sin(rax), cx = cos(rax)
          | -sz   cz   0    0 |     sy = sin(ray), cx = cos(ray)
          |  0    0    1    0 |     sz = sin(raz), cz = cos(raz)
          |  0    0    0    1 |

     R  = RX * RY * RZ  (Note: order is determined by rotateOrder)

     RX = |  1    0    0    0 |     RY = |  cy   0   -sy   0 |
          |  0    cx   sx   0 |          |  0    1    0    0 |
          |  0   -sx   cx   0 |          |  sy   0    cy   0 |
          |  0    0    0    1 |          |  0    0    0    1 |

     RZ = |  cz   sz   0    0 |     sx = sin(rx), cx = cos(rx)
          | -sz   cz   0    0 |     sy = sin(ry), cx = cos(ry)
          |  0    0    1    0 |     sz = sin(rz), cz = cos(rz)
          |  0    0    0    1 |

     T  = |  1    0    0    0 |
          |  0    1    0    0 |
          |  0    0    1    0 |
          |  tx   ty   tz   1 |
'''
__author__ = ('David Dunn')
__version__ = '2.0'

from math import sin, cos, tan, radians, degrees, atan2, asin
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as linalg

def eye(value):
    return matlib.identity(value)

def dtype():
    return type(eye(4))

def calcTransform(**kwargs):
    ''' Use the given component matricies to compose a full transform
    From Maya documentation:
         -1                       -1
      [Sp]x[S]x[Sh]x[Sp]x[St]x[Rp]x[Ro]x[R]x[Rp]x[Rt]x[T]
    where 'x' denotes matrix multiplication and '-1' denotes matrix inversion
    '''
    sp = kwargs.get('scalePivot',           kwargs.get('sp', matlib.identity(4)))
    s  = kwargs.get('scale',                kwargs.get('s',  matlib.identity(4)))
    sh = kwargs.get('sheer',                kwargs.get('sh', matlib.identity(4)))
    st = kwargs.get('scalePivotTranslate',  kwargs.get('st', matlib.identity(4)))
    rp = kwargs.get('rotatePivot',          kwargs.get('rp', matlib.identity(4)))
    ro = kwargs.get('rotateOrient',         kwargs.get('ro', matlib.identity(4)))
    r  = kwargs.get('rotate',               kwargs.get('r',  matlib.identity(4)))
    rt = kwargs.get('rotatePivotTranslate', kwargs.get('rt', matlib.identity(4)))
    t  = kwargs.get('translate',            kwargs.get('t',  matlib.identity(4)))
    return sp.getI()*s*sh*sp*st*rp.getI()*ro*r*rp*rt*t
    #return t*rt*rp*r*ro*rp.getI()*st*sp*sh*s*sp.getI()

def calcTranslate(translate):
    ''' Generate a translation component matrix from a given translation
       T  = |  1    0    0    0 |
            |  0    1    0    0 |
            |  0    0    1    0 |
            |  tx   ty   tz   1 |'''
    x,y,z=translate
    return np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[x,y,z,1]])

def calcOrient(orient):
    ''' Generate a orientation component matrix from a given orientation 
    Ro = AX * AY * AZ '''
    return calcRotation(orient, [0,1,2])

def calcRotation(rotation, ro):
    ''' Generate a rotation component matrix from a given rotation and rotation order
    From Maya documentation:
    R  = RX * RY * RZ  (Note: order is determined by rotateOrder)

    RX = |  1    0    0    0 |     RY = |  cy   0   -sy   0 |
         |  0    cx   sx   0 |          |  0    1    0    0 |
         |  0   -sx   cx   0 |          |  sy   0    cy   0 |
         |  0    0    0    1 |          |  0    0    0    1 |

    RZ = |  cz   sz   0    0 |     sx = sin(rx), cx = cos(rx)
         | -sz   cz   0    0 |     sy = sin(ry), cx = cos(ry)
         |  0    0    1    0 |     sz = sin(rz), cz = cos(rz)
         |  0    0    0    1 | '''
    x,y,z=rotation
    xm = np.matrix([[1,0,0,0],[0,cos(radians(x)),sin(radians(x)),0],[0,-sin(radians(x)),cos(radians(x)),0],[0,0,0,1]])
    ym = np.matrix([[cos(radians(y)),0,-sin(radians(y)),0],[0,1,0,0],[sin(radians(y)),0,cos(radians(y)),0],[0,0,0,1]])
    zm = np.matrix([[cos(radians(z)),sin(radians(z)),0,0],[-sin(radians(z)),cos(radians(z)),0,0],[0,0,1,0],[0,0,0,1]])
    # now put them in order accoring to rotateOrder
    mash = {}
    mash[ro[0]] = xm
    mash[ro[1]] = ym
    mash[ro[2]] = zm
    return mash[0]*mash[1]*mash[2]
    #return mash[2]*mash[1]*mash[0]

def calcQuaternion(quaternion):
    x,y,z,w=quaternion
    row0 = [1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),0]
    row1 = [2*(x*y-z*w),1-2*(x*x+z*z),2*(z*y+x*w),0]
    row2 = [2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y),0]
    return np.matrix([row0,row1,row2,[0,0,0,1]])

def calcScale(scale):
    '''Generate a scale component matrix from a given scale 
       S  = |  sx   0    0    0 |     
            |  0    sy   0    0 |     
            |  0    0    sz   0 |     
            |  0    0    0    1 |'''
    x,y,z=scale
    return np.matrix([[x,0,0,0],[0,y,0,0],[0,0,z,0],[0,0,0,1]])

def calcSheer(sheer):
    '''Sh = |  1    0    0    0 |
            | shxy  1    0    0 |
            | shxz shyz  1    0 |
            |  0    0    0    1 |'''
    xy,xz,yz=sheer
    return np.matrix([[1,0,0,0],[xy,1,0,0],[xz,yz,1,0],[0,0,0,1]])

def calcPerspective(n, f):
    # Not updated
    ''' Camera perspective transformation - n: near clip plane - f: far clip plane '''
    return np.matrix([[n,0,0,0],[0,n,0,0],[0,0,n+f,1],[0,0,-f*n,0]])

def calcPerspGL(fov, aspect, near, far):
    ''' Camera perspective transformation - n: near clip plane - f: far clip plane '''
    focal=1./tan(radians(fov)/2.)
    return np.matrix([[float(focal)/aspect,0,0,0],[0,focal,0,0],[0,0,(far+near)/(near-far),-1],[0,0,(2*far*near)/(near-far),0]])

def calcOrthoGL(r, l, t, b, n, f):
    ''' Camera orthographic projection transformation - define the frustum - r:right - l:left - t:top - b:bottom - n:near - f:far '''
    return np.matrix([[2./(r-l),0,0,0],[0,2./(t-b),0,0],[0,0,-2./(f-n),0],[-(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1]])

def calcProjection(r, l, t, b, n, f):
    # Not updated
    ''' Camera orthographic projection transformation - define the frustum - r:right - l:left - t:top - b:bottom - n:near - f:far '''
    return np.matrix([[2./(r-l),0,0,0],[0,2./(t-b),0,0],[0,0,2./(n-f),0],[-(r+l)/(r-l),-(t+b)/(t-b),-(n+f)/(n-f),1]])

def calcScreen(nx, ny):
    # Not updated
    ''' Transform to screen space - nx: resolution in x - ny: resolution in y'''
    return np.matrix([[nx/2.,0,0,0],[0,ny/2.,0,0],[0,0,1,0],[(nx-1)/2.,(ny-1)/2.,0,1]])
        
def _norm(vector):
    return vector/linalg.norm(vector)
def getTranslate(mat):
    return np.array(mat[3])[0][:3]
def getScale(mat):
    xVec = np.array(mat[0])[0][:3]
    yVec = np.array(mat[1])[0][:3]
    zVec = np.array(mat[2])[0][:3]
    # Remove sheer
    xNormVec = _norm(xVec)
    yNormVec = _norm(yVec)
    zCrossNormVec = _norm(np.cross(xNormVec,yNormVec))
    yCrossNormVec = np.cross(zCrossNormVec,xNormVec)
    xCrossNormVec = np.cross(yCrossNormVec,zCrossNormVec)   
    # Scale is the difference between the unit vector and the regular vector
    scale = np.array([0,0,0])
    scale[0] = np.dot(xCrossNormVec,xVec)
    scale[1] = np.dot(yCrossNormVec,yVec)
    scale[2] = np.dot(zCrossNormVec,zVec)
    return scale
def getSheer(mat):
    yVec = np.array(mat[1])[0][:3]
    zVec = np.array(mat[2])[0][:3]
    xNormVec = _norm(np.array(mat[0])[0][:3])
    xyCrossNormVec = _norm(np.cross(xNormVec,_norm(yVec)))
    xzCrossNormVec = np.cross(xyCrossNormVec,xNormVec)
    yzCrossNormVec = np.cross(xNormVec,xzCrossNormVec)
    shearXY = np.dot(xNormVec,yVec) / np.dot(xzCrossNormVec,yVec)
    shearXZ = np.dot(xNormVec,zVec) / np.dot(xyCrossNormVec,zVec)
    shearYZ = np.dot(xzCrossNormVec,zVec) / np.dot(yzCrossNormVec,zVec)
    return np.array([shearXY, shearXZ, shearYZ])
def getRotation(mat, ro):
    # Remove any sheer and scale first
    xNormVec = _norm(np.array(mat[0])[0][:3])
    yNormVec = _norm(np.array(mat[1])[0][:3])
    zCrossNormVec = _norm(np.cross(xNormVec,yNormVec))
    yCrossNormVec = np.cross(zCrossNormVec,xNormVec)
    xCrossNormVec = np.cross(yCrossNormVec,zCrossNormVec)
    crossList = [xCrossNormVec, yCrossNormVec, zCrossNormVec]
    rotation = np.array((0.,0.,0.))
    sgn = 1 if (ro[0]+1)%3 == ro[1] else -1  # check for right or left hand
    rotation[ro[0]] = degrees(atan2(sgn*crossList[ro[1]][ro[2]], crossList[ro[2]][ro[2]]))
    rotation[ro[1]] = degrees(asin(max(-1.0, min(1.0, -1*sgn*crossList[ro[0]][ro[2]]))))  # clamp for floating point error
    rotation[ro[2]] = degrees(atan2(sgn*crossList[ro[0]][ro[1]], crossList[ro[0]][ro[0]]))
    return rotation
def setOrient(mat, orient):
    pass
def setRotation(mat, rotation):
    pass
def setTranslate(mat, translate):
    pass
def setScale(mat, scale):
    pass
    
