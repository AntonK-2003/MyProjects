def Newton(x0, f, f1, e):
    #f1 - производная
    while True:
        x1 = x0 - (f(x0) / f1(x0))
        if abs(x1 - x0) < e:
            return x1
        x0 = x1
        
def dichotomy():
    print("Введите исходные данные: ")
    print("a = ", end='')
    a = float(input())
    print("b = ", end='')
    b = float(input())
    print("eps = ", end='')
    e = float(input())
    print("Вы ввели: ")
    print("a = %.2f  b = %.2f  eps = %.2e" % (a, b, e))
    
    y = log(a) - a + 1.8
    while b-a >= e:
        x = (a+b)/2
        z = log(x) - x + 1.8
        if y*z < 0:
            b = x
        else:
            a = x
            y = z
    
    print("x =", x, "z =",z)
    
def simple_iterations(x):
    return (2-0.4*x**2)**0.5+math.cos(x)
 
x1=float(input("Введите приближенное значение Х="))
e=float(input("Введите точность e="))
a=float(input("a="))
b=float(input("b="))
n=int(input("n="))
a=abs((fun(a+0.0001)-fun(a))/0.0001)
b=abs((fun(b+0.0001)-fun(b))/0.0001)
q=max(a,b)
q=(1-q)/q
 
iters=0
x0=x1
x1=fun(x0)
while abs(x1-x0) <= abs(q*e):
    iters+=1
    x0=x1
    x1=fun(x0)
print('Точное значение корня:',2.0926)
print('Вычисленное значение корня:',x1)
print('Число итераций:',iters)

def F(x):
    return 0.1 * math.pow(x, 2) - x * math.log(x)
 
 
 
def F1(x):
    res=0.2 * x - math.log(x) - 1
    print(res)
    return res
 
 
 
def secant_method(a, b):
    try:
        x0 = (a + b) / 2
        xn = F(x0)
        xn1 = xn - F(xn) / F1(xn)
        while abs(xn1 - xn) > math.pow(10, -5):
            xn = xn1 
            xn1 = xn - F(xn) / F1(xn)
        print(xn1)
        return xn1
    except ValueError:
        print("Value not invalidate")
if __name__ == '__main__':
    x=float(input())
    a=float(input())
    b=float(input())
    F(x)
    F1(x)
    Method(a, b)
