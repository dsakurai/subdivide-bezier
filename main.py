# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import torch
import torch_bsf
import plotly.express as px
import time

def make_w(num):
    w1 = 100
    list = []
    for _ in range(101): #len([0,1,2,...,100])=101
        w2 = 100 - w1
        w3 = 0
        for _ in range(101):
            #if w1 + w2 + w3 == 100 and w2 >= 0 and w1 >= 0 and w2 < w1*4+3:  #*num/(100 - num):
            #if w1 + w2 + w3 == 100 and w2 >= 0 and w1 >= 0 and w2 >= w1*4+3:  #*num/(100 - num):
            if w1 + w2 + w3 == 100 and w2 >= 0 and w1 >= 75:
            #if w1 + w2 + w3 == 100 and w2 >= 0 and w1 >= 0 and w2 <= w1*num/(100 - num):
                list2 = []
                list2.append(w1/100)
                list2.append(w2/100)
                list2.append(w3/100)
                list.append(list2)
            w2 -= 1
            w3 += 1
        w1 -= 1
    df = pd.DataFrame(list)
    df.to_csv("testdata.csv",header=False, index=False, sep="\t")

def calc_alpha(w0, eps):
    if w0 == 0:
        w0 = 0.01 # minimum w0
    return (1 - w0 + eps) / w0

def calc_L1_ratio(w0, w1, eps):
    b = w1 / (1 - w0 + eps)
    return b

def trans_param():
    data = np.loadtxt("testdata.csv")
    list = []
    for i in data:
        list2 = []
        e = 1e-16
        alpha = calc_alpha(i[0], e)
        L1_ratio = calc_L1_ratio(i[0], i[1], e)
        list2.append(alpha)
        list2.append(L1_ratio)
        list.append(list2)

    regcoef_df = pd.DataFrame(list)
    regcoef_df.to_csv("regcoef.csv",header=False, index=False, sep="\t")

def calc_EN():
    data_set_x = np.loadtxt("datax.csv")
    data_set_y = np.loadtxt("datay.csv")
    data_x = pd.DataFrame(data_set_x)
    data_y = pd.DataFrame(data_set_y)
    regcoef = np.loadtxt("regcoef.csv")
    list = []
    for i in regcoef:
        if i[1] < 1 - 1e-4:
            i[1] += 1e-4
        #print(i[0], i[1])
        elastic_net = ElasticNet(alpha = i[0] + 0.04, l1_ratio = i[1])  #li_ratio alpha 足さないとエラーが出る。 /Users/zaizenkiichi/PycharmProjects/pythonProject2/venv/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.473e+00, tolerance: 5.000e-04 Linear regression models with null weight for the l1 regularization term are more efficiently fitted using one of the solvers implemented in sklearn.linear_model.Ridge/RidgeCV instead.model = cd_fast.enet_coordinate_descent(
        elastic_net = elastic_net.fit(data_x, data_y)
        elastic_net_np = elastic_net.coef_.round(3)
        elastic_net_list = elastic_net_np.tolist()
        list.append(elastic_net_list)

    df = pd.DataFrame(list)
    df.to_csv("elastic_net.csv",header=False, index=False, sep="\t")

def f3(coef):
    X = np.array(coef)
    return np.linalg.norm(X, ord = 2)**2

def f1c(data_x, data_y, coef):
    #calc 1/2M||X0 - y||^2
    #np.matmul(data_x, coef.T) = X0
    #(np.matmul(data_x, coef.T) - data_y) = X0 - y
    M = 0
    for _ in data_x:
        M += 1
    g = (np.linalg.norm((np.matmul(data_x, coef.T) - data_y).T, ord = 2)**2)/(2*M)
    return g + 1e-16 * f3(coef)

def f2c(coef):
    #calc |0|
    X = np.array(coef)
    return np.linalg.norm(X, ord = 1) + 1e-16 * f3(coef)

def f3c(coef):
    #calc 1/2||0||^2
    X = np.array(coef)
    return (np.linalg.norm(X, ord = 2)**2)/2 + 1e-16 * f3(coef)

def calc_PF():
    data_set_x = np.loadtxt("datax.csv")
    data_set_y = np.loadtxt("datay.csv")
    elasticnet_coef = np.loadtxt("elastic_net.csv")
    list = []
    for i in elasticnet_coef:
        list2 = []
        a = f1c(data_set_x, data_set_y, i)
        b = f2c(i)
        c = f3c(i)

        d = a.tolist()
        e = b.tolist()
        f = c.tolist()
        list2.append(d)
        list2.append(e)
        list2.append(f)
        list.append(list2)

    df = pd.DataFrame(list)
    df.to_csv("pareto_front.csv",header=False, index=False, sep="\t")

def make_data_file():
    pareto_set = np.loadtxt("elastic_net.csv")
    pareto_front = np.loadtxt("pareto_front.csv")
    list = []
    pareto_set_list = pareto_set.tolist()
    pareto_front_list = pareto_front.tolist()
    for i in range(len(pareto_set_list)):
        list2 = []
        for j in range(len(pareto_set_list[0])):
            list2.append(pareto_set_list[i][j])
        for k in range(3):
            list2.append(pareto_front_list[i][k])
        list.append(list2)
    df = pd.DataFrame(list)
    df.to_csv("dataf123.csv",header=False, index=False, sep="\t")

def bs_fit(d):
    ts = np.loadtxt("testdata.csv")
    xs = np.loadtxt("dataf123.csv")

    ts = torch.tensor(ts)
    xs = torch.tensor(xs)

    bs = torch_bsf.fit(params=ts, values=xs, degree=d)

    #t = [[0.2, 0.3, 0.5]]
    #x = bs(t)
    #print(f"{t} -> {x}")

    xdf = pd.DataFrame(xs[:, 3:6], columns=['f1','f2','f3'])
    pf = xdf.values.tolist()

    _, bts = bs.meshgrid(num=100)
    bts = bts.detach()
    df = pd.DataFrame(bts[:, 3:6],columns=['f1','f2','f3'])
    bezier = df.values.tolist()
    e = 0
    m = 0
    print(len(pf))
    print(len(bezier))
    for i in range(len(pf)):
        a = (bezier[i][0] - pf[i][0])**2
        b = (bezier[i][1] - pf[i][1])**2
        c = (bezier[i][2] - pf[i][2])**2
        d = np.sqrt(a + b + c)
        e += d
        if m < d:
            m = d

    e = e/len(pf)

    fig = px.scatter_3d(xdf, x='f1', y='f2', z='f3')
    #fig.show()
    fig = px.scatter_3d(df, x='f1', y='f2', z='f3')
    fig.show()

    return e, m

cut_rate = 80
Mode = 0 # Mode = 0 not cut, 1 cut left, 2 cut right
start_time = time.perf_counter()

make_w(cut_rate)
trans_param()
calc_EN()
calc_PF()
make_data_file()

Llist_ave = []
Llist_max = []
Llist_time = []

for i in range(1):
    list_ave = []
    list_max = []
    list_time = []
    for j in range(15):
        d = j + 1
        start = time.perf_counter()
        e, m = bs_fit(d)
        end = time.perf_counter()
        tm = end - start
        list_ave.append(e)
        list_max.append(m)
        list_time.append(tm)
    Llist_ave.append(list_ave)
    Llist_max.append(list_max)
    Llist_time.append(list_time)
adf = pd.DataFrame(Llist_ave)
mdf = pd.DataFrame(Llist_max)
tdf = pd.DataFrame(Llist_time)
adf.to_csv("ave_err.csv",header=False, index=False, sep="\t")
mdf.to_csv("max_err.csv",header=False, index=False, sep="\t")
tdf.to_csv("calc_time.csv",header=False, index=False, sep="\t")
end_time = time.perf_counter()
print(end_time - start_time)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
