import pandas as pd
import numpy as np

# Отримання даних з ексель файлу
table = pd.read_excel("lab4.xlsx")
prod = table.to_numpy()
print("Вхідні дані")
print(prod)
print("\n\n")
prod = np.delete(prod, np.s_[0], axis=1)

number_prod = 8
number_criteria = 12


# Нормування факторів
def min_norming(row):
    sum = 0
    for k in range(number_prod):
        sum += prod[row, k]
    return 1 / sum


def max_norming(row):
    sum = 0
    for k in range(number_prod):
        sum += 1 / prod[row, k]
    return 1 / sum


normed_criteria = np.zeros((number_criteria, number_prod))

for i in range(number_criteria):
    sum = 0
    if prod[i, 8] == "min":
        sum += min_norming(i)
    else:
        sum += max_norming(i)

    for j in range(number_prod):
        if prod[i, 8] == "min":
            normed_criteria[i, j] = prod[i, j] * sum
        else:
            normed_criteria[i, j] = sum / prod[i, j]

# Нормування вагових коефіцієнтів
assessments = np.zeros((number_prod))

coefficients = [2, 2, 1, 1.5, 1, 1, 1.3, 1.3, 1, 1, 1, 1]

sum = 0
for i in range(number_criteria):
    sum += coefficients[i]

for i in range(number_criteria):
    coefficients[i] /= sum

# Отримання інтегральних оцінок для користувачів
def multicriteria_assessment(crits):
    assessment = 0
    for i in range(number_criteria):
        assessment += coefficients[i] / (1 - crits[i])
    return assessment


for i in range(number_prod):
    assessments[i] = multicriteria_assessment(normed_criteria[:, i])


# Пошук мінімальної оцінки та виведення результатів
optimal_product = assessments.argmin()

print("Оцінки ефективності товарів")
for i in range(number_prod):
    print(table.columns[i], ": ", assessments[i])

print("\n")
print("Індекс найоптимальнішого товару: ", optimal_product)
print("Найоптимальніший товар для випуску: ", table.columns[optimal_product+1])
