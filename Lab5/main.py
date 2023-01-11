# -------- Лекція 11 побудова штучної нейронної мережі для прогнозування трендів з tensorflow ---------
# ---------------------------- ПРИКЛАД 3: прогнозування ліній тренду ----------------------------------
import math
import random

import tensorflow._api.v2.compat.v1 as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# оголошення версії tensorflow
tf.disable_v2_behavior()

# --------------------------------------------------------------------------------------------------
m = 4000
n = 500
dm = 0
dsig = 10  # параметри закону розподілу ВВ із систематикою dsig
S = ((np.random.randn(n, m)) * dsig) + dm

part = 15
n_abnorm = int(n * part / 100)
err = np.random.normal(0, dsig, n)

sign = [1, -1]
for i in range(n):
    err[i] *= random.choice(sign)

abnorm_indexes = random.sample(range(n), n_abnorm)

data_train = np.zeros((m, n))
data_test = np.zeros((m, n))
abnorm = np.zeros((m, n))

for j in range(n):
    abnorm_indexes = random.sample(range(m), n_abnorm)
    for i in abnorm_indexes:
        S[j, i] += np.random.normal(0, dsig*3, 1)

for j in range(m):
    for i in range(n):
        data_train[j, i] = (1500*math.cos(0.005*j)) + S[i, j]  # квадратична модель реального процесу
        data_test[j, i] = (1700 * math.cos(0.002 * j)) + S[i, j]

plt.plot(data_train)
plt.plot(data_test)
plt.ylabel('динаміка тестів')
plt.show()

print('------- data_train---------')
print(data_train)
print('------- data_test---------')
print(data_test)
# ----------------------------------------------------------------------------------------------------

# масштабування (нормалізація) даних для діапазона значень -1, 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

print('------- data_train_norm ---------')
print(data_train)
print('------- data_test_norm ---------')
print(data_test)

# побудова простору даних x, y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# кількість запитів з навчальної вибірки
n_stocks = X_train.shape[1]

# параметри нейромережі за прошарками - визначаються у т.ч. структурою вхідних даних
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Ініциалізація процедур розрахунку
net = tf.InteractiveSession()

X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# моделювання прошарків мережі - з використанням методів tensorflow !!!
# скритий рівень вагових коефіціентів
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# вихідний рівень вагових коефіціентів
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# скритий рівень (прошарок) мережі
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# вихідний рівень (прошарок) мережі
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# функція помилки - ініціалізує блок навчання за критерієм - mean squared error (MSE)
mse = tf.reduce_mean(tf.squared_difference(out, Y))
# оптимізація - знаходження мінімуму (алгоритм мінімізації - градієнтний)
opt = tf.train.AdamOptimizer().minimize(mse)
# ініціалізація пошуку мінімуму
net.run(tf.global_variables_initializer())

# графічні відображення результатів обрахунку
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# вихідні параметри для сегментів тренування нейронної мережі
batch_size = 20
mse_train = []
mse_test = []

# запуск на виконання
epochs = 4  # кількість епох навчання визначає якість апроксимації та прогнозу
for e in range(epochs):

    # формування даних для навчання
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # навчання на сегментах вибірки
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # відображення динаміки навчання та прогнозування
        if np.mod(i, 50) == 0:
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])

            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))  # графічні відображення динаміки результатів обрахунку
            plt.pause(0.0002)

print('------- pred---------')
print(pred)
