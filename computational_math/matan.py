from math import sin
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, cos, sin

def rectangle_method(f, a, b, n):
    print("\nТекущее число разбиений: ", n)
    h = (b-a)/float(n)
    print("Текущий шаг:", h)
    total = sum([f((a + (k*h))) for k in range(0, n)])
    result = h * total
    print("Текущий результат: ", result)
    return result

def trapezium_method(func, 0, pi, delta):
    def integrate(func, mim_lim, max_lim, n):
        integral = 0.0
        step = ( 0, pi) / n
        for x in range( 0, pi-step, step):
            integral += step*(func(x) + func(x + step)) / 2
        return integral
 
    d, n = 1, 1
    while math.fabs(d) > delta:
        d = (integrate(func,  0, pi, n * 2) - integrate(func,  0, pi, n)) / 3
        n *= 2
 
    a = math.fabs(integrate(func,  0, pi, n))
    b = math.fabs(integrate(func,  0, pi, n)) + d
    if a > b:
        a, b = b, a
    print('Trapezium:')
    print('\t%s\t%s\t%s' % (n, a, b))

def sympson_method(left, right, n, function):
    h = (right - left) / (2 * n)
 
    tmp_sum = float(function.subs({x: left})) +\
        float(function.subs({x: right}))
 
    for step in range(1, 2 * n):
        if step % 2 != 0:
            tmp_sum += 4 * float(function.subs({x: left + step * h}))
        else:
            tmp_sum += 2 * float(function.subs({x: left + step * h}))
 
    return tmp_sum * h / 3

def interpolation(xp,fp):
    res = interp(xp,fp)
    return res
    #  Точки интерполируемой функции:
    xp = np.linspace(-np.pi, np.pi, 21)
    fp = np.sinc(xp)

    #  Вычисленные точки интерполяции:
    x = np.linspace(-np.pi, np.pi, 61)
    y = np.interp(x, xp, fp)

    fig, ax = plt.subplots()

    ax.plot(xp, fp,
            marker = 'o',
            label = 'sinc(x)')
    ax.plot(x, y,
            marker = 'x',
            label = 'interp')

    ax.set_title('Линейная интерполяция точек')

    plt.show()

def derivative(F):
    res = diff(F)
    return res

def lim(F, var, p):
    res = limit(F, var, p)

def function_graph(x, F):
    plt.plot(x, F)
    plt.show()
    
