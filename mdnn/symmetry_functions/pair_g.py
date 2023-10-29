class PairG(object):
    """
    Class stores values of g and dg for one atom
    """
    def __init__(self, g, dg):
        self.g = g
        self.dg = dg

    def __add__(self, pair): 
        self.g += pair.g
        for i in range(3):
            self.dg[i] += pair.dg[i]
        return self

    def __repr__(self) -> str:
        return f"PairG(g: {self.g}, dg: {self.dg})"