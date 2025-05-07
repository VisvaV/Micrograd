import math
import numpy as np
import matplotlib.pyplot as plt
from graph import *

class Value:
    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value = {self.data}"
    
    def __add__(self, other):
        out = Value(self.data+other.data, (self, other), '+')
        return out
        
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(-3.0, label = 'a')
b = Value(5.0, label = 'b')
c = Value(10.0, label = 'c')
e = a*b ; e.label = 'e'
d = e + c ; d.label = 'd'
f = Value(6.0, label = 'f')
L = d * f ; L.label = 'L'

dot = draw_dot(L)
dot.render('graph_output', format='png', view=True)

