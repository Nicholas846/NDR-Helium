class BasisFunction:
    def __init__(self, zeta, l, m):
        self.zeta = float(zeta)
        self.l = int(l)
        self.m = int(m)

    def __repr__(self):
        return f"BasisFunction(zeta={self.zeta}, l={self.l}, m={self.m})"


def make_basis(*zeta_lists):
    bf = []

    for l, zetas in enumerate(zeta_lists):
        for zeta in zetas:
            for m in range(-l, l + 1):
                bf.append(BasisFunction(zeta, l, m))
    return bf
