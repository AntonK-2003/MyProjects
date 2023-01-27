def Euler(n = 10, h = 0.01, x = 1, y = 1, f):
    for i in range(n):
        y += h * f(x, y)
        x += h
    return x, y # решение

def runge_kutta(y, x, dx, f):
    k1 = dx * f(y, t)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

