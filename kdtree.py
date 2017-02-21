#!/usr/bin/env python
'''kdTree implementation - read only at this point

David Dunn
Jan 2017 - created by splitting off from dGraph

ALL UNITS ARE IN METERS 
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.6'

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