import math
import copy
from functools import reduce
import numpy as np
def adicion_veccom (*vectores):
    suma_real = 0
    suma_img = 0
    for vector in vectores:
        suma_real += vector[0]
        suma_img += vector [1]
    return [suma_real,suma_img]

def inverso_veccom (v1):
    return [(v1[0]*-1),(v1[1]*-1)]

def multi_esca_veccom (n1,v1):
    return [(n1*v1[0]),(n1*v1[1])]

def imprimir_bonito_matriz(m):
    for fila in m:
        print("[", end=" ")
        for elemento in fila:
            print(elemento, end=" ")
        print("]")

def add_matrizcom (m1,m2):
    if len(m1) == len(m2) and len(m1[0]) == len(m2[0]):
        m3 = []
        for i in range(len(m1)):
            m3.append([])
            for j in range(len(m1[0])):
                m3[i].append(adicion_veccom (m1[i][j] , m2[i][j]))
        return m3
    else:
        return None

a= [
    [[1,0],[2,-1],[0,3]],
    [[-1,0],[3,0],[0,0]],
    [[1,-1],[2,2],[0,2]]
]

b= [
    [[-1,0],[-2,0],[-3,0]],
    [[-6,0],[-8,0],[-4,1]],
    [[-1,-3],[0,1],[1,-1]]
]

#imprimir_bonito_matriz(add_matrizcom(a,b))

def transpuesta_ma (m):
    t=[]
    for columna in range(len(m[0])):
        t.append([])
        for fila in range(len(m)):
            t[columna].append(m[fila][columna])
    return t
#imprimir_bonito_matriz(transpuesta_ma(a))

def conjugada_vector (v1):
    return [(v1[0]),(v1[1]*-1)]

def conjugada_matriz(m):
    con = []
    for fila in range(len(m)):
        con.append([])
        for columna in range(len(m[0])):
            con[fila].append(conjugada_vector(m[fila][columna]))
    return con
#imprimir_bonito_matriz(conjugada_matriz(a))

def multiplicacion_escalar_matriz (n1,m):
    mul = []
    for fila in range(len(m)):
        mul.append([])
        for columna in range(len(m[0])):
            mul[fila].append(multi_esca_veccom(n1,m[fila][columna]))
    return mul
#imprimir_bonito_matriz(multiplicacion_escalar_matriz(2,a))

def producto_matrices (m1,m2):
    if len(m1[0])==len(m2):
        res = []
        for fila in range(len(m1)):
            res.append([])
            for columna in range(len(m2[0])):
                res[fila].append(0)
        for fila in range(len(m1)):
            for columna in range(len(m2[0])):
                for k in range(len(m1[0])):
                    res[fila][columna] += m1[fila][k] * m2[k][columna]
        return res
    return None
f = [
    [1,2,3],
    [4,5,6]
]
g = [
    [1,2],
    [3,4],
    [5,6]
]
o= [
    [1,-3],
    [1,2],
    [2,0],
    [-1,1]
]
u= [-1,-4,-0,-2]
#imprimir_bonito_matriz(producto_matrices(u,o))

def vec_producto_matriz (v1,m1):
    if len(v1) == len(m1):
        res = []
        sum1 = 0
        sum2 = 0
        for fila in range(len(v1)):
            res.append([])
            for columna in range(len(m1[0])):
                res[fila].append(v1[fila]*m1[fila][columna])
        for fila in res:
            sum1 += fila[0]
            sum2 += fila[1]
        return [sum1, sum2]
    return None
#print(vec_producto_matriz(u,o))

def producto_interno (v1,v2):
    return (v1[0]*v2[0])+(v1[1]*v2[1])
#print(producto_interno([-3,5],[2,-4]))

def norma_vector (v1):
    return math.sqrt(math.pow(v1[0],2)+math.pow(v1[1],2))
#print(norma_vector([3,4]))

def distancia_entre_vectores (v1,v2):
    return norma_vector([v2[0]-v1[0],v2[1]-v1[1]])
#print(distancia_entre_vectores([1,-2],[3,4]))

h=[
    [[5,0],[3,7]],
    [[3,-7],[2,0]]
]

def comparar_matrices(m1, m2):
    for fila in range(len(m1)):
        for columna in range(len(m2[0])):
            if (m1[fila][columna] != m2[fila][columna]):
                return False
    return True

def matriz_hermetica (m1):
    return comparar_matrices(transpuesta_ma(conjugada_matriz(m1)),m1)
#print(matriz_hermetica(a))

def sum_com(n1,n2):
    return [n1[0]+n2[0],n1[1]+n2[1]]

def pro_com(n1, n2):
    real = (n1[0] * n2[0]) - (n1[1] * n2[1])
    ima = (n1[0] * n2[1]) + (n1[1] * n2[0])
    return [real, ima]
#print(pro_com([0,3],[1,-1]))

def multiplicacion_escalar_matrizimg (v1,m):
    mul = []
    for fila in range(len(m)):
        mul.append([])
        for columna in range(len(m[0])):
            mul[fila].append(pro_com(v1,m[fila][columna]))
    return mul

def pro_tensor (m1,m2):
    total = []
    for fila in range(len(m1)):
        total.append([])
        for columna in range(len(m1[0])):
            total[fila].append(multiplicacion_escalar_matrizimg(m1[fila][columna],m2))
    return total
#imprimir_bonito_matriz(pro_tensor(h,a))

def producto_matricescom (m1,m2):
    if len(m1[0])==len(m2):
        res = []
        for fila in range(len(m1)):
            res.append([])
            for columna in range(len(m2[0])):
                res[fila].append([0,0])
        for fila in range(len(m1)):
            for columna in range(len(m2[0])):
                for k in range(len(m1[0])):
                    res[fila][columna] = sum_com(res[fila][columna], pro_com(m1[fila][k], m2[k][columna]))
        return res
    return None

#imprimir_bonito_matriz(producto_matricescom(a,b))

y=[
    [[0,1],[0,0],[0,0]],
    [[0,0],[0,1],[0,0]],
    [[0,0],[0,0],[0,1]]
]

def matriz_identidadcom (n1):
    resulado = []
    for fila in range(n1):
        resulado.append([])
        for columna in range(n1):
            if fila == columna:
                resulado[fila].append([1,0])
            else:
                resulado[fila].append([0,0])
    return resulado
#imprimir_bonito_matriz(matriz_identidadcom(3))

def matriz_unitaria (m1):
    return comparar_matrices(producto_matricescom(m1, transpuesta_ma(conjugada_matriz(m1))),matriz_identidadcom(len(m1)))
#print(matriz_unitaria(y))


     #ADJUNTA
def remove_line(indice, v1):
    result = []
    for i in range(len(v1)):
        adj = []
        for j in range(len(v1)):
            if (i == indice[0] or j == indice[1]):
                pass
            else:
                adj.append(v1[i][j])
        if (adj != []):
            result.append(adj)
    # yy.remove([])
    return result


def prod_adjunta(v1,v2,negative):
    if negative:
        return (-((v1[0] * v2[1]) - (v2[0] * v1[1])))
    return ((v1[0]*v2[1])-(v2[0]*v1[1]))
#print(prod_adjunta([3,2],[4,5],True))

def por_adjunta(v1,negative):
    return prod_adjunta(v1[0],v1[1],negative)

def adjunta(m1):
    if len(m1)== 3 and len(m1[0]) == 3:
        respuesta = []
        for fila in range(len(m1)):
            respuesta.append([])
            for columna in range(len(m1[0])):
                temporal = remove_line(tuple([fila,columna]),m1)
                if (fila + columna) % 2 == 0:
                    respuesta[fila].append(por_adjunta(temporal,False))
                else:
                    respuesta[fila].append(por_adjunta(temporal,True))
        return respuesta
    return None

b=[
    [-3,2,0],
    [1,-1,2],
    [-2,1,3]
]

z=[
    [2,-2,2],
    [2,1,0],
    [3,-2,2]
]
w = [[1,2,4],[3,4,7],[5,6,7]]

#imprimir_bonito_matriz(adjunta(b))

def det(matrix):
    order=len(matrix)
    posdet=0
    for i in range(order):
        posdet+=reduce((lambda x, y: x * y), [matrix[(i+j)%order][j] for j in range(order)])
    negdet=0
    for i in range(order):
        negdet+=reduce((lambda x, y: x * y), [matrix[(order-i-j)%order][j] for j in range(order)])
    return posdet-negdet
print(det(w))















def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

imprimir_bonito_matriz(inverse(z))