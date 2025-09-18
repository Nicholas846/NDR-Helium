class BasisFunction:
    def __init__(self, zeta, l, m):
        self.zeta = float(zeta)
        self.l = int(l)
        self.m = int(m)

    def __repr__(self):
        return f"BasisFunction(zeta={self.zeta}, l={self.l}, m={self.m})"

def make_basis(zeta_list, l_list):
    bf = []
    for zeta in zeta_list:
        for l in l_list:
            for m in range(-l, l + 1):
                bf.append(BasisFunction(zeta, l, m))
    return bf

