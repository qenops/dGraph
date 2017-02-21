#!/usr/bin/env python
'''Shape submodule for dGraph scene description module

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METRIC 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'
__all__ = ["Shape", "PolySurface"]

from dGraph import *
import dGraph.materials as dgm
from dGraph.io import obj
from math import sin, cos, pi
import numpy as np
from numpy.linalg import norm
from numpy import dot, cross, matlib

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
        self.setMaterial(dgm.Material('default'))
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
