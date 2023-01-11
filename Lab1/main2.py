import numpy as np
import matplotlib.pyplot as plt
import math

# Модель генерації випадкової величини - похибки вимірювання та визначення їхніх характеристик
n = 10000
sig = 10
loc = 0
err = np.random.normal(loc, sig, n)

med = np.median(err)
var = np.var(err)
dev = math.sqrt(var)

print('Статистичні характеристики похибки')
print('Медіана похибки вимірювання', med)
print('Дисперсія похибки вимірювання', var)
print('Середньоквадратичне відхилення похибки вимірювання', dev, '\n')

plt.hist(err, bins=20)
plt.title('Похибка вимірювань')
plt.show()


# Модель зміни досліджуваного процесу
trend = np.empty(n)
a = 0.03
b = -100
for i in range(n):
    trend[i] = (a * i + b)

plt.plot(trend)
plt.title('Тренд вимірюваного процесу')
plt.show()


# Модель експериментального вимірювання та визначення їхніх статистичних характеристик
experimental = np.empty(n)

for i in range(n):
    experimental[i] = trend[i] + err[i]

exp_med = np.median(experimental)
exp_var = np.var(experimental)
exp_dev = math.sqrt(exp_var)

print('Статистичні характеристики експериментальних даних')
print('Медіана експериментальних значень', exp_med)
print('Дисперсія експериментальних значень', exp_var)
print('Середньоквадратичне відхилення експериментальних значень', exp_dev)

plt.title('Експериментальні значення')
plt.plot(experimental)
plt.plot(trend, color='yellow')
plt.show()


# Гістограми експериментальних даних та похибки вимірювання
plt.hist(err, label='Похибка вимірювання', color='red', alpha=0.3)
plt.hist(experimental, label='Експериментальні дані', color='green', alpha=0.3)
plt.title('Гістограми похибки вимірювання та експериментальних даних')
plt.show()

