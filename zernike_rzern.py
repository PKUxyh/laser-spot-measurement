import numpy as np
from scipy.special import factorial

class RZern:
    def __init__(self, n):
        self.n = n
        self.coeffs = None

    def make_cart_grid(self, r, theta):
        self.r = r
        self.theta = theta

    def zernike_radial(self, n, m, r):
        R = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            R += r ** (n - 2 * k) * (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k)
            )
        return R

    def zernike(self, n, m, r, theta):
        if m > 0:
            return self.zernike_radial(n, m, r) * np.cos(m * theta)
        elif m < 0:
            return self.zernike_radial(n, -m, r) * np.sin(-m * theta)
        else:
            return self.zernike_radial(n, 0, r)

    def fit(self, data):
        self.coeffs = []
        mask = self.r <= 1
        for n in range(self.n + 1):
            for m in range(-n, n + 1, 2):
                Z = self.zernike(n, m, self.r[mask], self.theta[mask])
                c = (data[mask] * Z).sum() / (Z ** 2).sum()
                self.coeffs.append(c)
        return np.array(self.coeffs)

    def eval_grid(self, coeffs, matrix=False):
        result = np.zeros_like(self.r)
        idx = 0
        for n in range(self.n + 1):
            for m in range(-n, n + 1, 2):
                Z = self.zernike(n, m, self.r, self.theta)
                result += coeffs[idx] * Z
                idx += 1
        if matrix:
            return np.where(self.r <= 1, result, 0)
        return result
