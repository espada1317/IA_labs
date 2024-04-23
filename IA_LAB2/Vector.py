import math


class Vector:
    def __init__(self, components):
        self.components = components

    def norm(self):
        return math.sqrt(sum(comp ** 2 for comp in self.components))

    def __add__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension for addition.")
        return Vector([x + y for x, y in zip(self.components, other.components)])

    def __sub__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension for subtraction.")
        return Vector([x - y for x, y in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([comp * scalar for comp in self.components])

    def __truediv__(self, scalar):
        if scalar == 0:
            raise ValueError("Cannot divide by zero.")
        return Vector([comp / scalar for comp in self.components])

    def dot_product(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension for dot product.")
        return sum(x * y for x, y in zip(self.components, other.components))

    def cross_product(self, other):
        if len(self.components) != 3 or len(other.components) != 3:
            raise ValueError("Cross product is defined only for 3-dimensional vectors.")
        x = self.components[1] * other.components[2] - self.components[2] * other.components[1]
        y = self.components[2] * other.components[0] - self.components[0] * other.components[2]
        z = self.components[0] * other.components[1] - self.components[1] * other.components[0]
        return Vector([x, y, z])
