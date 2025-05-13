import numpy as np
import random as r
import json
import math

application= 10 ** 6  # заявки (m)
attribute= 30 # атрибуты (n)
p= 80  # порог точности 80%
alfa= 0.05   #скорость обучения (если происходит хорошее обучение, а потом отупление имеет смысл его понизить)
iteration= 0
max_iteration= 1000

def GenerateDataSet():
    X = []
    Y = []
    countDef = 0
    countNotDef = 0
    for _ in range(application):
        count = 0
        x_row = []
        for _ in range(attribute):
            val = r.randint(0, 10)
            x_row.append(val)
            if 7<=val<=10:
                count += 1

        y = 1 if count > 10 else 0
        Y.append(y)
        X.append(x_row)
        countDef+= y
        countNotDef += (1 - y)

    print("Число дефолтов:", countDef,
          "Число не дефолтов:", countNotDef)
    with open("data.json", "w") as f:
        json.dump({"X": X, "Y": Y}, f)

def Sigmoid(y):
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y))
    else:
        e = math.exp(y)
        return e / (1.0 + e)

def Check_res(Y_res, Y):
    global iteration
    good_ans = 0
    for i in range(len(Y)):
        pred = 1 if Y_res[i] >= 0.5 else 0
        if Y[i][0] == pred:
            good_ans += 1

    ans = good_ans / len(Y) * 100
    print(f"Итерация: {iteration}  Точность: {ans:.2f}%\n")

    if iteration > 0 and ans >= p:
        print("Обучение закончено\n")
        return True
    elif iteration >= max_iteration:
        print("Превышен лимит итераций\n")
        return True

    iteration += 1
    return False

def dJdW(Y_res, Y, X, j):
    res = 0.0
    for i in range(application):
        res += (Y_res[i] - Y[i][0]) * X[i][j]
    return res / application

def dJdB(Y_res, Y):
    res = 0.0
    for i in range(application):
        res += (Y_res[i] - Y[i][0])
    return res / application

def Study(Y_res, Y, W, b, X):
    for j in range(attribute):
        grad_w = dJdW(Y_res, Y, X, j)
        W[j][0] -= alfa * grad_w
    grad_b = dJdB(Y_res, Y)
    b -= alfa * grad_b
    return W, b

def main():
    #GenerateDataSet()

    with open("data.json", "r") as f:
        data = json.load(f)

    X = np.array(data["X"], dtype=np.float32)
    X = X / 10.0 # нормирую X, чтобы он не так сильно различался с весами

    Y = np.array(data["Y"], dtype=np.float32).reshape(-1, 1)
    b = -0.5  # bias

    W = []
    for _ in range(attribute):
        W.append(r.uniform(-0.01, 0.01)) #заполнение в диапозоне (-0.01, 0.01)
    W = np.array(W, dtype=np.float32).reshape(-1, 1)

    while True:
        Z = np.dot(X, W) + b
        Y_res = [Sigmoid(z[0]) for z in Z]

        if Check_res(Y_res, Y):
            with open("answer.json", "w") as f:
                json.dump({"best_W": W.tolist(), "best_B": b}, f)
            break

        W, b = Study(Y_res, Y, W, b, X)

if __name__ == "__main__":
    main()