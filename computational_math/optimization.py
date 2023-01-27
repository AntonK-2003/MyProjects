import numpy as np
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
 
def gradsteps(f, r, epsg=1e-6, alpha=0.4, maxiter=100): 
    #xlist = [r]
    for itera in range(maxiter):
        arr_r = dfdx(r) 
        summa = sum(arr_r**2)
        r = r + alpha * arr_r 
        if np.sqrt(summa) < epsg:
            break      
    r = np.round(r, 6)
    return r.tolist(), f(r), itera
 
r = np.array([-10, -10])
 
gradsteps_f_x = gradsteps(f, r)
print(gradsteps_f_x)

# Формирование сетки
X = np.arange(-2, 2, 0.1)
Y = np.arange(-1.5, 3, 0.1)
X, Y = np.meshgrid(X, Y)
# Функция Розенброка
Z = (1.0 - X)**2 + 100.0 * (Y - X * X)**2
#
fig = plt.figure()
# Будем выводить 3d-проекцию графика функции
ax = fig.gca(projection = '3d')
# Вывод поверхности
surf = ax.plot_surface(X, Y, Z, cmap = cm.Spectral, linewidth = 0, antialiased = False)
# Метки осей координат
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Настройка оси X
for label in ax.xaxis.get_ticklabels():
    label.set_color('black')
    label.set_rotation(-45)
    label.set_fontsize(9)
# Настройка оси Y
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(9)
# Настройка оси Z
for label in ax.zaxis.get_ticklabels():
    label.set_fontsize(9)
# Изометрия
ax.view_init(elev = 30, azim = 45)
# Шкала цветов
fig.colorbar(surf, shrink = 0.5, aspect = 5)
# Отображение результата (рис. 1)
plt.show()

# Функция Розенброка
def Rosenbrock(X):
    return (1.0 - X[0])**2 + 100.0_8 * (X[1] - X[0] * X[0] )**2
#
n = 2
x0 = np.zeros(2, dtype = float) # Вектор с двумя элементами типа float
# Начальная точка поиска минимума функции
x0[0] = -5.0
x0[1] = 10.0
xtol = 1.0e-5 # Точность поиска экстремума
# Находим минимум функции
res = opt.minimize(Rosenbrock, x0, method = 'Nelder-Mead', options = {'xtol': xtol, 'disp': True})
print(res)

def method(x1,x2,h,eps):
#Проверка на содержание корня
    if f(x1)*f(x2)>0:
        return None,None
    else:
#Проверка на содержание корня в одной из заданных точек
        if f(x1) == 0 or f(x2) == 0:
            if f(x1) == 0:
                xk=x1
                intt=0
                return xk,intt
            if f(x2) == 0:
                xk=x2
                intt=0
                return xk,intt
 
        else:
#Вычисление кол-ва иттераций и поиск корня
            xk=0
            intt=0
            while abs(x1-x2)>eps:
                x3=x1+((3-sqrt(5))/2)*(x2-x1)
                x4=x1+((sqrt(5)-1)/2)*(x2-x1)
                if abs(f(x3))>abs(f(x4)):
                    x1=x3
                else:
                    x2=x4
                intt+=1
            xk=x3
            return xk,intt

exec ('z = lambda a: ' + z_str)

z_str = z_str.replace('a[0]', 'x')
z_str = z_str.replace('a[1]', 'y')

def z_grad(a):
    x = Symbol('x')
    y = Symbol('y')

    z_d = eval (  z_str) #exec ('z_d =  ' + z_str)

    yprime = z_d.diff(y)
    dif_y=str(yprime).replace('y', str(a[1]))
    dif_y=dif_y.replace('x', str(a[0]))

    yprime = z_d.diff(x)
    dif_x=str(yprime).replace('y', str(a[1]))
    dif_x=dif_x.replace('x', str(a[0]))

    return numpy.array([eval(dif_y), eval(dif_x)])


def mininize(a):
    l_min = minimize_scalar(lambda l: z(a - l * z_grad(a))).x
    return a - l_min * z_grad(a)

def norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)

def grad_step(dot):
    return mininize(dot)

dot = [numpy.array([-150.0, 150.0])]
dot.append(grad_step(dot[0]))

eps = 0.0001

while norm(dot[-2] - dot[-1]) > eps: dot.append(grad_step(dot[-1]))
def makeData ():
    x = numpy.arange (-200, 200, 1.0)
    y = numpy.arange (-200, 200, 1.0)
    xgrid, ygrid = numpy.meshgrid(x, y)
    zgrid = z([xgrid, ygrid])
    return xgrid, ygrid, zgrid

xt, yt, zt = makeData()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xt, yt, zt, cmap=cm.hot)
ax.plot([x[0] for x in dot], [x[1] for x in dot], [z(x) for x in dot], color='b')

plt.show()

def unlin_prog(x, y, z, f):
    x,y,z=symbols(' x y z' )
    fx=f.diff(x)
    fy=f.diff(y)
    fz=f.diff(z)
    sols=solve([fx,fy,fz],x,y,z)
    fxx=f.diff(x,x).subs({x:sols[x],y:sols[y],z:sols[z]})
    fxy=f.diff(x,y).subs({x:sols[x],y:sols[y],z:sols[z]})
    fxz=f.diff(x,z).subs({x:sols[x],y:sols[y],z:sols[z]})
    fyx=f.diff(y,x).subs({x:sols[x],y:sols[y],z:sols[z]})
    fyy=f.diff(y,y).subs({x:sols[x],y:sols[y],z:sols[z]})
    fyz=f.diff(y,z).subs({x:sols[x],y:sols[y],z:sols[z]})
    fzx=f.diff(z,x).subs({x:sols[x],y:sols[y],z:sols[z]})
    fzy=f.diff(z,y).subs({x:sols[x],y:sols[y],z:sols[z]})
    fzz=f.diff(z,z).subs({x:sols[x],y:sols[y],z:sols[z]})
    d1=fxx
    M2=Matrix([[fxx,fxy],[fyx,fyy]])
    d2=M2.det()
    M3=Matrix([[fxx,fxy,fxz],[fyx,fyy,fyz],[fzx,fzy,fzz]])
    d3=M3.det()
    if  d1>0 and d2>0 and d3>0:
            print('При d1=%s,>0, d2=%s>0, d3=%s>0, минимум f в точке М(%s,%s,%s)'%(str(d1),str(d2),str(d3),str(sols[x]),str(sols[y]),str(sols[z])))
    elif  d1<0 and d2>0 and d3<0:
            print('При d1=%s,<0, d2=%s>0, d3=%s<0,максимум f в точке М(%s,%s,%s)'%(str(d1),str(d2),str(d3),str(sols[x]),str(sols[y]),str(sols[z])))
    elif  d3!=0:
            print('Седло в точке М(%s,%s,%s)'%(str(sols[x]),str(sols[y]),str(sols[z])))
    else:
            print('Нет экстремума в точке М(%s,%s,%s)'%(str(sols[x]),str(sols[y]),str(sols[z])))
    r=f.subs({x:sols[x],y:sols[y],z:sols[z]})          
            print('Значение %s функции в точке М(%s,%s,%s)'%(str(r),str(sols[x]),str(sols[y]),str(sols[z])))

def conditional_opt(a, b, c, d, condition):
    bounds = Bounds ([a, b], [c, d])
    linear_constraint = LinearConstraint ([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
    
def unconditional_opt():


def lin_prog():

    
