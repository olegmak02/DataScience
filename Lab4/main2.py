import pandas as pd
import numpy as np

# Отримання даних з ексель файлу
table = pd.read_excel("lab4.xlsx", sheet_name="main2")
prod = table.to_numpy()
print("Вхідні дані")
print(prod)
print("\n\n")
prod = np.delete(prod, np.s_[0], axis=1)
number_user = 12
number_criteria = 9

# Нормування факторів
def min_norming(row):
    sum = 0
    for k in range(number_user):
        sum += prod[row, k]
    return 1 / sum


def max_norming(row):
    sum = 0
    for k in range(number_user):
        sum += 1 / (prod[row, k] + 0.1)
    return 1 / sum


normed_criteria = np.zeros((number_criteria, number_user))

for i in range(number_criteria):
    sum = 0
    if prod[i, 12] == "min":
        sum += min_norming(i)
    else:
        sum += max_norming(i)

    for j in range(number_user):
        if prod[i, 12] == "min":
            normed_criteria[i, j] = prod[i, j] * sum
        else:
            normed_criteria[i, j] = sum / prod[i, j]


# Нормування вагових коефіцієнтів
assessments = np.zeros((number_user))

coefficients = [1.8, 1, 1.5, 1.5, 1, 1.4, 1, 1.2, 1.4]

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


for i in range(number_user):
    assessments[i] = multicriteria_assessment(normed_criteria[:, i])

# Пошук мінімальної оцінки та виведення результатів
optimal_user = assessments.argmin()

print("Оцінки користувачів")
for i in range(number_user):
    print(table.columns[i], ": ", assessments[i])

print("\n")
print("Індекс користувача з найкращою оцінкою: ", optimal_user)
print("Користувач з найкращою оцінкою: ", table.columns[optimal_user+1])
