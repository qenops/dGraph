#!/usr/bin/env python
'''An OBJ reader and writer

Builds a mesh with multiple lists: verts and faces at a minimum
verts is a list of arrays each defining the position of each vert
uvs is a list of arrays each defining the position of a uv coordinate
normals is a list of arrays each defining a vector of a normal
faces is a list of dicts each defining indicies in the vertex, uvs, and normals list which together create a face

David Dunn
Sept 2014 - Created
www.qenops.com

http://paulbourke.net/dataformats/obj/

The following types of data may be included in an .obj file. In this list, the keyword (in parentheses) follows the data type.

Vertex data
o       geometric vertices (v)
o       texture vertices (vt)
o       vertex normals (vn)
o       parameter space vertices (vp)
o	    Free-form curve/surface attributes rational or non-rational forms of curve or surface type: basis matrix, Bezier, B-spline, Cardinal, Taylor (cstype)
o       degree (deg)
o       basis matrix (bmat)
o       step size (step)

Elements
o       point (p)
o       line (l)
o       face (f)
o       curve (curv)
o       2D curve (curv2)
o       surface (surf)

Free-form curve/surface body statements
o       parameter values (parm)
o       outer trimming loop (trim)
o       inner trimming loop (hole)
o       special curve (scrv)
o       special point (sp)
o       end statement (end)

Connectivity between free-form surfaces
o       connect (con)

Grouping
o       group name (g)
o       smoothing group (s)
o       merging group (mg)
o       object name (o)

Display/render attributes
o       bevel interpolation (bevel)
o       color interpolation (c_interp)
o       dissolve interpolation (d_interp)
o       level of detail (lod)
o       material name (usemtl)
o       material library (mtllib)
o       shadow casting (shadow_obj)
o       ray tracing (trace_obj)
o       curve approximation technique (ctech)
o       surface approximation technique (stech)
'''
__author__ = ('David Dunn')
__version__ = '1.0'

import numpy as np
from numpy.linalg import norm
import itertools as it

def load(file, normalize=False):
    ''' Load an OBJ from a file '''
    verts = []
    uvs = []
    normals = []
    faceVerts = []
    faceUvs = []
    faceNormals = []
    #minV = [float('inf'),float('inf'),float('inf')]
    #maxV = [-float('inf'),-float('inf'),-float('inf')]
    with open(file) as f:
        for line in f:
            if len(line) <= 1:
                continue
            tokens = line.split()
            if tokens[0] == '#':  # Comment
                pass
            elif tokens[0] == 'v':  # List of Vertices, with (x,y,z[,w]) coordinates, w is optional and defaults to 1.0.
                v = map(float, tokens[1:4])
                #minV = map(min, v, minV)
                #maxV = map(max, v, maxV)
                if len(tokens) > 4:
                    v.append(float(tokens[5]))
                verts.append(v)
            elif tokens[0] == 'vt':  # Texture coordinates, in (u, v [,w]) coordinates, these will vary between 0 and 1, w is optional and defaults to 0.
                uv = map(float, tokens[1:3])
                if len(tokens) > 3:
                    uv.append(float(tokens[4]))
                uvs.append(uv)
            elif tokens[0] == 'vn':  # Normals in (x,y,z) form; normals might not be unit.
                n = map(float, tokens[1:4])
                if normalize:
                    n/=norm(n)
                normals.append(n)
            elif tokens[0] == 'vp':  # Parameter space vertices in ( u [,v] [,w] ) form; free form geometry statement.
                pass
            elif tokens[0] == 'mtllib':  # external .mtl file name
                pass
            elif tokens[0] == 'usemtl':  # specifies the material name for the element following it.
                pass
            elif tokens[0] == 'o':  # object name
                pass
            elif tokens[0] == 'g':  # group name
                pass
            elif tokens[0] == 's':  # smoothing group
                pass
            elif tokens[0] == 'p':  # a point in a mesh
                pass
            elif tokens[0] == 'l':  # a line in a mesh
                pass
            elif tokens[0] == 'f':  # face definitions - can have 0,1 or 2 slashes defining vertex/uv/normal
                ''' if vertex numbers are negative they refer to the verts listed previously - boo stupid standard, glad no one uses that - I'm not supporting it '''
                vertList = []
                uvList = []
                normList = []
                for tok in tokens[1:]:
                    v = tok.split('/')
                    vertList.append(int(v[0])-1)
                    if len(v) > 1 and v[1] != '':
                        uvList.append(int(v[1])-1) 
                    if len(v) > 2 and v[2] != '':
                        normList.append(int(v[2])-1)
                #faces.append({'verts':vertList,'uvs':uvList,'normals':normList})
                faceVerts.append(vertList)
                faceUvs.append(uvList) 
                faceNormals.append(normList)
            elif tokens[0] == 'curv':  # a curve
                pass
            elif tokens[0] == 'curv2': # a surface curve
                pass
            elif tokens[0] == 'surf':  # a surface
                pass
    
    verts, uvs, normals = [np.matrix(a,dtype=np.float32) if a != [] else np.matrix([]) for a in [verts, uvs, normals]]
    faceSizes = map(len, faceVerts)
    if max(faceSizes) == min(faceSizes):    # convert to matrix since all faces are same size
        faceVerts, faceUvs, faceNormals = [np.matrix(a,dtype=np.uint32) if a != [] else np.matrix([]) for a in [faceVerts, faceUvs, faceNormals]]
    #else:                                   # convert to single array???? - this seems stupid
    #    faceVerts, faceUvs, faceNormals = [np.array([i for sub in a for i in sub],dtype=np.uint32) if a != [] else [] for a in [faceVerts, faceUvs, faceNormals]]
    print("Loaded mesh from %s. (%d vertices, %d uvs, %d normals, %d faces)"%(file, len(verts), len(uvs), len(normals), len(faceVerts)))
    #print "Mesh bounding box is: (%0.4f, %0.4f, %0.4f) to (%0.4f, %0.4f, %0.4f)"%(minV[0], minV[1], minV[2], maxV[0], maxV[1], maxV[2])
    return verts, uvs, normals, faceVerts, faceUvs, faceNormals, faceSizes

def write(file, verts, uvs, normals, faceVerts, faceUvs, faceNormals ):
    ''' Write out an obj to a file '''
    output = ''
    with open(file,'w') as f:
        output = ''.join([('v %s\n'%' '.join([str(i) for i in v])) for v in verts.A.tolist()])
        f.write('%s\n'%output)
        if len(uvs.A1) > 2:
            output = ''.join([('vt %s\n'%' '.join([str(i) for i in uv])) for uv in uvs.A.tolist()])
            f.write('%s\n'%output)
        if len(normals.A1) > 2:
            output = ''.join([('vn %s\n'%' '.join([str(i) for i in n])) for n in normals.A.tolist()])
            f.write('%s\n'%output)
        output = ''.join([('f %s\n'%' '.join([('/'.join([str(i) for i in v])) for v in zip(a,b,c)])) for a,b,c in it.izip_longest((faceVerts+1).A.tolist(), (faceUvs+1).A.tolist(), (faceNormals+1).A.tolist(), fillvalue=['','',''])])
        output = output.split('\n',1)[1]
        f.write('%s\n'%output)
        
#if __name__ == '__main__':
    #verts, uvs, tempNormals, faces = loadObj(sys.argv[1])
    #writeObj('new_%s'%sys.argv[1], verts, uvs, tempNormals, faces)