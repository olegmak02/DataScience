import numpy as np
import matplotlib.pyplot as plt
import random
import math

n = 100

# Модель зміни досліджуваного процесу
trend = np.empty(n)
for i in range(n):
    trend[i] = (0.03 * i * i)

# Модель генерації випадкової величини - похибки вимірювання та визначення їхніх характеристик
scale = 10
part = 10
n_abnorm = int(n * part / 100)
err = np.random.normal(0, scale, n)

sign = [1, -1]

for i in range(n):
    err[i] *= random.choice(sign)  # Нормальний розподіл з додатніми та від'ємними значеннями

exp_clear = []
for i in range(n):
    exp_clear.append(trend[i] + err[i])

med = np.median(exp_clear)
var = np.var(exp_clear)
dev = math.sqrt(var)

print('медіана вибірки без АВ')
print(med)
print('дисперсія вибірки без АВ')
print(var)
print('СКВ вибірки без АВ')
print(dev)
print('\n\n\n')

plt.hist(exp_clear, bins=20)
plt.title('Гістограма вибірки без АВ')
plt.show()

med = np.median(err)
var = np.var(err)
dev = math.sqrt(var)

print('медіана похибки')
print(med)
print('дисперсія похибки')
print(var)
print('СКВ похибки')
print(dev)
print('\n\n\n')

plt.hist(err, bins=20)
plt.title('Гістограма похибки')
plt.show()

# Модель генерації аномальної випадкової величини
abnorm_indexes = random.sample(range(n), n_abnorm)

for i in abnorm_indexes:
    err[i] += np.random.normal(0, scale * 3, 1)

# Модель експериментального вимірювання та визначення їхніх статистичних характеристик
experimental = np.empty(n)

for i in range(n):
    experimental[i] = trend[i] + err[i]

med = np.median(experimental)
var = np.var(experimental)
dev = math.sqrt(var)

print('медіана вибірки з АВ')
print(med)
print('дисперсія вибірки з АВ')
print(var)
print('СКВ вибірки з АВ')
print(dev)
print('\n\n\n')

plt.hist(experimental, bins=20)
plt.title('Гістограма вибірки з АВ')
plt.show()

plt.plot(experimental)
plt.show()


# Виявлення аномальних вимірів

def LS(Y, F):
    Ft = F.T
    FtF = Ft.dot(F)
    FtFI = np.linalg.inv(FtF)
    FtFTIFt = FtFI.dot(Ft)
    C = FtFTIFt.dot(Y)
    T = F.dot(C)
    return T.flatten()


def recovery(index):
    return (experimental[index - 1] + experimental[index + 1]) / 2


def recovery2(index):
    return (2 * experimental[index - 1] + experimental[index + 2]) / 3


size_wnd = 50
start = 0

calc_errs = np.zeros(size_wnd)
Y = np.zeros((size_wnd, 1))
F = np.ones((size_wnd, 3))

subset = experimental[0:size_wnd]

for i in range(size_wnd):
    Y[i, 0] = float(subset[i])
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)

T = LS(Y, F)

for i in range(size_wnd):
    calc_errs[i] = experimental[i] - T[i]

mediana = np.median(calc_errs)
disp = np.var(calc_errs)
sigma = math.sqrt(disp)
cor_ind = []

for i in range(size_wnd):
    if abs(calc_errs[i]) > 3 * sigma:
        cor_ind.append(i)

for i in range(len(cor_ind) - 1):
    if cor_ind[i + 1] - cor_ind[i] == 1 and cor_ind[i] < n - 1:
        experimental[start + cor_ind[i]] = recovery2(start + cor_ind[i])
    else:
        if cor_ind[i] >= n - 1:
            experimental[start + cor_ind[i]] = experimental[start + cor_ind[i] - 1]
        else:
            experimental[start + cor_ind[i]] = recovery(start + cor_ind[i])
if len(cor_ind) > 0:
    experimental[cor_ind[len(cor_ind) - 1]] = recovery(cor_ind[len(cor_ind) - 1])

for i in range(size_wnd):
    calc_errs[i] = experimental[i] - T[i]

mediana = np.median(calc_errs)
disp = np.var(calc_errs)
etalon = math.sqrt(disp)
prev_sigma = etalon
start = 0
window = []

while start < n - size_wnd - 1:
    for i in range(len(T)):
        window.append(experimental[start + i])
    window.remove(window[0])
    window.append(experimental[start + size_wnd + 1])

    Y = np.zeros((size_wnd, 1))
    F = np.ones((size_wnd, 3))
    for i in range(size_wnd):
        Y[i, 0] = float(experimental[i + start])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    T = LS(Y, F)
    c = []
    for i in range(size_wnd):
        c.append(experimental[start + i] - T[i])

    for i in range(len(T)):
        window[i] = experimental[start + i] - T[i]

    sigma = math.sqrt(np.var(window))
    if abs(T[len(T) - 1] - experimental[start + len(T) - 1]) > 3 * prev_sigma:
        experimental[start + len(T) - 1] = recovery(start + len(T) - 1)

    calc_errs = []
    for i in range(size_wnd):
        calc_errs.append(experimental[i + start] - T[i])

    sigma = math.sqrt(np.var(calc_errs))
    start += 1

plt.plot(experimental)
plt.show()

# Згладжування експериментальної вибірки за допомогою фільтру Калмана (матрична форма)
T0 = 1

F = np.zeros((3, 3))

F[0, 0] = 1
F[1, 1] = 1
F[2, 2] = 1
F[0, 1] = T0
F[0, 2] = T0 * T0 / 2
F[1, 2] = T0

H = np.zeros((1, 3))
H[0, 0] = 1

G = np.zeros((2, 1))
G[0, 0] = T0 * T0 * T0 * T0 / 24
G[1, 0] = T0 * T0 * T0 / 6


def pred(F, X, P):
    X_new = F.dot(X)

    s = (G.T).dot(G)
    P_new = F.dot(P.dot(F.T)) + s
    return X_new, P_new


def cor(H, X, Z, P):
    Y = Z - H.dot(X)
    S = H.dot(P.dot(H.T))
    PHt = P.dot(H.T)
    K = PHt / (S[0, 0] + sigma)
    X_new = np.zeros((3, 1))
    X_new[0, 0] = X[0, 0] + K[0, 0] * Y[0, 0]
    HK = K[0,0] * H[0,0] + K[1,0] * H[0,1] + K[2,0] * H[0,2]
    P_new = P * (1 - HK)
    return X_new, P_new


P = np.zeros((3, 3))
P_prev = np.zeros((3, 3))
P[0, 0] = sigma*sigma
P[1, 1] = sigma*sigma
P[2, 2] = sigma*sigma
P_prev[0, 0] = sigma*sigma
P_prev[1, 1] = sigma*sigma
P_prev[2, 2] = sigma*sigma

i = 3
arr = []
arr.append(experimental[0])
arr.append(experimental[1])

X_prev = np.zeros((3, 1))
X_prev[0, 0] = experimental[2]
X_prev[1, 0] = (experimental[2] - experimental[1]) / 2
X_prev[2, 0] = (experimental[2] - 2 * experimental[1] - experimental[0]) / 4
X = np.zeros((3, 1))

while i < n:
    X_prev, P_prev = pred(F, X_prev, P_prev)
    X[0, 0] = experimental[i]
    X[1, 0] = (experimental[i] - experimental[i-1]) / 2
    X[2, 0] = (experimental[i] - 2 * experimental[i-1] - experimental[i-2]) / 4
    X_prev, P_prev = cor(H, X_prev, X, P_prev)
    arr.append(X_prev[0, 0])
    i += 1

med = np.median(arr)
var = np.var(arr)
dev = math.sqrt(var)

print('медіана результатів згладжування фільтром Калмана')
print(med)
print('дисперсія результатів згладжування фільтром Калмана')
print(var)
print('СКВ результатів згладжування фільтром Калмана')
print(dev)
print('\n\n\n')

plt.hist(arr, bins=20, alpha=0.5)
plt.hist(experimental, bins=20, alpha=0.5)
plt.hist(exp_clear, bins=20, alpha=0.5)
plt.title('Гістограма результатів')
plt.show()

plt.plot(experimental, color='green')
plt.plot(exp_clear, color='black', alpha=0.5)
plt.plot(arr, color='blue')
plt.plot(trend, color='red', alpha=0.5)
plt.show()
