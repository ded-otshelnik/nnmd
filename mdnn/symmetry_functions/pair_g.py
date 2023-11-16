class PairG(object):
    """
    Class stores value of g and derivatives by radius-vectors and params for one atom 
    """
    def __init__(self, g_type, g, dg):
        self.g_type = g_type
        self.g = g
        self.dg = dg

    def __add__(self, pair): 
        self.g.append(pair.g)
        for i in range(3):
            self.dg[i] += pair.dg[i]
        return self

    def __repr__(self) -> str:
        return f"PairG(G{self.g_type}, g: {self.g}, dg: {self.dg}, dg_params: {self.dg_params})"