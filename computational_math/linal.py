def print_matrix( A ):
    for strA in A:
        print( strA )
        
def minor( A, i, j ):
    M = copy.deepcopy(A)  # копирование!
    del M[ i ]
    for i in range( len( A[0] ) - 1 ):
        del M[ i ] [ j ]
    return M   
    
def det( A ):
    m = len( A )
    n = len( A[0] )
    if m != n:
        return None
    if n == 1:
        return A[0][0]
    signum = 1
    determinant = 0
    # разложение по первой строке
    for j in range( n ):
        determinant += A[0][j]*signum*det( minor( A, 0, j ) ) 
        signum *= -1
    return determinant 

def Cramer(A,B):
    m = len(A)
    op = np.linalg.det(A)
    r = list()
    for i in range(m):
        VM = np.copy(A)
        VM[:,i] = B
        r.append(np.linalg.det(VM)/op)
    return r
    

def Gauss(matrix):
    for nrow in range(len(matrix)):
         pivot = nrow + np.argmax(abs(matrix[nrow:, nrow]))
         if pivot != nrow:
             matrix[[nrow, pivot]] = matrix[[pivot, nrow]]
             row = matrix[nrow]
             divider = row[nrow]
             if abs(divider) < 1e-10:
                  raise ValueError(f"Решений нет")
             row /= divider
             for lower_row in matrix[nrow+1:]:
                  factor = lower_row[nrow]
                  lower_row -= factor*row
                  make_identity(matrix)
                  return matrix

def Seidel(A, b, eps):
    n = len(A)
    x = np.zeros(n)  # zero vector

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = np.linalg.norm(x_new - x) <= eps
        x = x_new

    return x

def strassen_mul_2x2(lb, rb):
	d = strassen_mul(lb[0][0] + lb[1][1], rb[0][0] + rb[1][1])
	d_1 = strassen_mul(lb[0][1] - lb[1][1], rb[1][0] + rb[1][1])
	d_2 = strassen_mul(lb[1][0] - lb[0][0], rb[0][0] + rb[0][1])

	left = strassen_mul(lb[1][1], rb[1][0] - rb[0][0])
	right = strassen_mul(lb[0][0], rb[0][1] - rb[1][1])
	top = strassen_mul(lb[0][0] + lb[0][1], rb[1][1])
	bottom = strassen_mul(lb[1][0] + lb[1][1], rb[0][0])

	return [[d + d_1 + left - top, right + top],
			[left + bottom, d + d_2 + right - bottom]]


def isCorrectArray(a):
    for row in range(0, len(a)):
        if( len(a[row]) != len(b) ):
            print('Не соответствует размерность')
            return False
    
    for row in range(0, len(a)):
        if( a[row][row] == 0 ):
            print('Нулевые элементы на главной диагонали')
            return False
    return True

def isNeedToComplete(x_old, x_new):
    eps = 0.0001
    sum_up = 0
    sum_low = 0
    for k in range(0, len(x_old)):
        sum_up += ( x_new[k] - x_old[k] ) ** 2
        sum_low += ( x_new[k] ) ** 2
        
    return math.sqrt( sum_up / sum_low ) < eps

def solution(a, b):
    if( not isCorrectArray(a) ):
        print('Ошибка в исходных данных')
    else:
        count = len(b) # количество корней
        
        x = [1 for k in range(0, count) ] # начальное приближение корней
        
        numberOfIter = 0  # подсчет количества итераций
        MAX_ITER = 100    # максимально допустимое число итераций
        while( numberOfIter < MAX_ITER ):
 
            x_prev = copy.deepcopy(x)
            
            for k in range(0, count):
                S = 0
                for j in range(0, count):
                    if( j != k ): S = S + a[k][j] * x[j] 
                x[k] = b[k]/a[k][k] - S / a[k][k]
            
            if isNeedToComplete(x_prev, x) : # проверка на выход
                break
            
            numberOfIter += 1
 
        print('Количество итераций на решение: ', numberOfIter)
        
        return x
    


def Jacobi(mx,mr,n=100,c=0.0001):
         если len (mx) == len (mr): # Если mx и mr равны по длине, запустите итерацию, в противном случае уравнение не имеет решения
                 x = [] # Итеративное начальное значение, инициализированное одной строкой, все 0 матрицы
        for i in range(len(mr)):
            x.append([0])
                 count = 0 # Подсчитать количество итераций
        while count < n:
                         nx = [] # Сохранить набор значений после одной итерации
            for i in range(len(x)):
                nxi = mr[i][0]
                for j in range(len(mx[i])):
                    if j!=i:
                        nxi = nxi+(-mx[i][j])*x[j][0]
                nxi = nxi/mx[i][i]
                                 nx.append ([nxi]) # Итеративно вычислил следующее значение xi
                         lc = [] # хранить множество ошибок между результатами двух итераций
            for i in range(len(x)):
                lc.append(abs(x[i][0]-nx[i][0]))
            if max(lc) < c:
                                 return nx # Когда ошибка соответствует требованиям, вернуть результат расчета
            x = nx
            count = count + 1
                 print("Решений нет.") # Если заданный результат итерации все еще не удовлетворен, уравнение не имеет решения
    else:
        print("Решений нет.")
