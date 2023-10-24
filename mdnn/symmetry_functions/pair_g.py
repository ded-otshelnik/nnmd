class PairG(object):
    """
    Class stores values of g and dg for one atom
    """
    def __init__(self, g, dg):
        self.g = g
        self.dg = dg

    def __add__(self, pair) :
        self.g += pair.g
        self.dg += pair.dg

    @property
    def T(self):
        return [[self.g], [self.dg]]