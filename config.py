#!/usr/bin/env python
'''Configuration file for setting up dGraph globals

David Dunn
Jun 2017 - created

ALL UNITS ARE IN METRIC
    ie 1 cm = .01

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '0.1'

shaderHeader = '''
#version 330
#ifdef GL_ES
    precision highp float;
#endif
'''

maxFPS = 60