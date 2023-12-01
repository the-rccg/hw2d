class Namespace(dict):
    def __mul__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] * val for key, val in self.items()})
        else:
            return Namespace({key: other * val for key, val in self.items()})

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: val / other[key] for key, val in self.items()})
        else:
            return Namespace({key: val / other for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: val / other[key] for key, val in self.items()})
        else:
            return Namespace({key: val / other for key, val in self.items()})

    def __rdiv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] / val for key, val in self.items()})
        else:
            return Namespace({key: other / val for key, val in self.items()})

    def __add__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] + val for key, val in self.items()})
        else:
            return Namespace({key: other + val for key, val in self.items()})

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] - val for key, val in self.items()})
        else:
            return Namespace({key: other - val for key, val in self.items()})

    @property
    def omega(self):
        return self["omega"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def density(self):
        return self["density"]

    @property
    def dE(self):
        return self["dE"]

    @property
    def dU(self):
        return self["dU"]

    @property
    def age(self):
        return self["age"]

    @property
    def dtype(self):
        return self["density"].dtype

    def copy(self):
        return Namespace({key: val for key, val in self.items()})
