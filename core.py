# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from math import log10
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import ElasticNet
import time
import torch
import torch_bsf
import unittest
import plotly.express as px

class Subdivision:
    triangle_0      = 0
    triangle_1      = 1
    triangle_2      = 2
    triangle_center = 3

#TODO suspicious epsilon
eps = 0.0000000001

# Number of flips in the triangle
def flipped(upper_triangle):
    """
    Check if the upper_triangle is flipped or not.
    :param upper_triangle: 一つ上の階層の三角形
    :return: True / False
    """
    num_flips = upper_triangle.count(Subdivision.triangle_center)
    return num_flips % 2 == 1

def in_triangle_(smallest_triangle: [int], w: [float], c= 0):
    """
    Check if the point `w` is in the specified `triangle`.
    Presumption: in_triangle() for higher level all returns True.
    :param smallest_triangle: hierarchy of triangles
    :param w: barycentric coordinates of the point.
    :param c: 内部の三角形を見るときに、周りの三角形と共通する辺を含むようにする
    :return: True / False
    """

    if abs(sum(w) - 1.0 ) > eps: raise Exception("Sum of w coordinates is not 1.")

    if smallest_triangle == []:
        return 0.0 <= w[0] and 0.0 <= w[1] and 0.0 <= w[2]

    t = smallest_triangle[-1]  # triangle position
    # in a corner triangle?
    if t in [Subdivision.triangle_0, Subdivision.triangle_1, Subdivision.triangle_2]:
        # yes

        # The triangle in the upper level of the hierarchy
        upper_triangle = smallest_triangle[:-1]

        def boundary(
                level = 1, # hierarchical level of the triangle subdivision
                bnd = 0.5,  # return value (boundary of the check)
                smallest_triangle = smallest_triangle, # the smallest triangle
        ):
            """
                Threshold of w for user-specified level of the triangle subdivision.
                To decide whether w is in this corner triangle, we check whether w is larger (or smaller) than this boundary value.
            """

            # binary search inside [0, 1]
            if level == len(smallest_triangle): return bnd

            tri = smallest_triangle[level - 1]
            fl = flipped(smallest_triangle[:level-1])
            if tri == t: # check inside?
                # yes
                if not fl:
                    bnd += 1 / 2 ** (level + 1) # move left
                else:
                    bnd -= 1 / 2 ** (level + 1) # move right
            else: # no => check outside
                if not fl:
                    bnd -= 1 / 2 ** (level + 1) # move right
                else:
                    bnd += 1 / 2 ** (level + 1) # move left

            return boundary(level=level+1, bnd=bnd)

        if c != 1:
            if not flipped(upper_triangle): return w[t] >= boundary()
            else:                           return w[t] <= boundary()
        else:
            if not flipped(upper_triangle): return w[t] > boundary()
            else:                           return w[t] < boundary()
    else:
        if t != Subdivision.triangle_center : raise Exception("Bad triangle")
        return \
                (not in_triangle_(smallest_triangle=smallest_triangle[:-1] + [Subdivision.triangle_0], w=w,c= 1)) and \
                (not in_triangle_(smallest_triangle=smallest_triangle[:-1] + [Subdivision.triangle_1], w=w,c= 1)) and \
                (not in_triangle_(smallest_triangle=smallest_triangle[:-1] + [Subdivision.triangle_2], w=w,c= 1))

def in_triangle(triangle: [int], w: [float]):

    def outside (triangle, w):
        return not in_triangle_(triangle, w)

    levels = range(len(triangle) + 1)

    for lev in levels:
        if outside(triangle[0:lev], w):
            return False
    return True

def make_w(
        resolution: int = 100,
        triangle: [int] = [] # default: largest triangle
):
    """
    Return [(w1, w2, w3)].
    :param resolution: number of points along each edge of the (largest, i.e. global) triangle
    :param triangle: [...].
        The smaller triangle within the largest one. This is an enum from `Subdivision`.
        The largest triangle is [].
        The left corner of the first subdivision is [Subdivision.triangle_0]
        The right corner of the 2nd subdivision in the left corner of the 1st is [Subdivision.triangle_0, Subdivision.triangle_1],
        ...
    """
    w1 = resolution
    ls = []
    for _ in range(resolution+1): #len([0,1,2,...,100])=101
        w2 = resolution - w1
        w3 = 0
        for _ in range(resolution+1):
            if w1 + w2 + w3 == resolution and w1 >= 0 and w2 >= 0 and w3 >= 0:
                list2 = []
                list2.append(w1/resolution)
                list2.append(w2/resolution)
                list2.append(w3/resolution)
                i = 0
                while True:
                    if in_triangle_(smallest_triangle=triangle[:i],w=list2):
                        if i == len(triangle):
                            ls.append(list2)
                            break
                    else:
                        break
                    i+=1
            w2 -= 1
            w3 += 1
        w1 -= 1
    return ls

def trans(triangle, bnd, w):
    """
    参照三角形をベジエ単体近似に使えるようにパラメータを変換する。変換のための関数。
    :param triangle: 変換される三角形
    :param bnd: 変換される三角形の三辺の境界の値
    :param w: 変換される三角形上の座標
    :return: 変換された座標
    """
    tlist = []
    if len(triangle) == 0:
        return w
    fl = flipped(upper_triangle=triangle)
    if not fl:
        for i in w:
            tlist2 = []
            t1 = (i[0] - bnd[0])*(2 **len(triangle))
            t2 = (i[1] - bnd[1])*(2 **len(triangle))
            t3 = (i[2] - bnd[2])*(2 **len(triangle))
            tlist2.append(t1)
            tlist2.append(t2)
            tlist2.append(t3)
            tlist.append(tlist2)
    else:
        for i in w:
            tlist2 = []
            t1 = (bnd[0] - i[0])*(2 **len(triangle))
            t2 = (bnd[1] - i[1])*(2 **len(triangle))
            t3 = (bnd[2] - i[2])*(2 **len(triangle))
            tlist2.append(t1)
            tlist2.append(t2)
            tlist2.append(t3)
            tlist.append(tlist2)
    return tlist

def maket(
        w,
        triangle: [int] = [] # default: largest triangle
):
    """
    参照三角形をベジエ単体近似に使えるようにパラメータを変換する。変換する三角形の三辺の境界を見つけてtrans()を呼ぶ
    :param w: 変換される三角形上の座標
    :param triangle: 変換される三角形
    :return: 変換された座標
    """
    border = [0.0, 0.0, 0.0]
    if len(triangle) == 0:
        tlist = trans(triangle, border, w)
        df = pd.DataFrame(tlist)
        df.to_csv("datat123.csv",header=False, index=False, sep="\t")
        return tlist

    def boundary(
            level = 1, # hierarchical level of the triangle subdivision
            bnd = 0.5,  # return value (boundary of the check)
            triangle = triangle, # the smallest triangle
    ):
        """
            Threshold of w for user-specified level of the triangle subdivision.
            To decide whether w is in this corner triangle, we check whether w is larger (or smaller) than this boundary value.
        """

        in_t = triangle[-1]
        in_f_tri = triangle
        # binary search inside [0, 1]
        if level == len(triangle): return bnd

        tri = triangle[level - 1]
        fl = flipped(triangle[:level-1])
        if tri == in_t: # check inside?
            # yes
            if not fl:
                bnd += 1 / 2 ** (level + 1) # move left
            else:
                bnd -= 1 / 2 ** (level + 1) # move right
        else: # no => check outside
            if not fl:
                bnd -= 1 / 2 ** (level + 1) # move right
            else:
                bnd += 1 / 2 ** (level + 1) # move left

        return boundary(level=level+1, bnd=bnd, triangle=in_f_tri)
    i = 0
    while True:
        t = triangle[i]
        if t in [Subdivision.triangle_0, Subdivision.triangle_1, Subdivision.triangle_2]:
            border[t] = boundary(triangle=triangle[:i+1])
        else:
            border[0] = boundary(triangle=triangle[:i]+[Subdivision.triangle_0])
            border[1] = boundary(triangle=triangle[:i]+[Subdivision.triangle_1])
            border[2] = boundary(triangle=triangle[:i]+[Subdivision.triangle_2])
        if i == len(triangle) - 1:
            break
        i += 1
    print(border)
    tlist = trans(triangle, border, w)
    df = pd.DataFrame(tlist)
    df.to_csv("datat123.csv",header=False, index=False, sep="\t")
    return tlist

def calc_alpha(w0, eps):
    w0 = max(w0,0.01) # TODO This 0.01 avoids explosion of alpha, but is 0.01 a good choice?
    return (1 - w0 + eps) / w0

def calc_L1_ratio(w0, w1, eps):
    b = w1 / (1 - w0 + eps)
    return b

def trans_param(w):
    """
    参照三角形上の座標をエラスティックネットの計算用のハイパーパラメータに変換する
    :param w: 変換される座標
    :return: 変換されたハイパーパラメータ
    """
    ls = []
    for i in w:
        list2 = []
        e = 1e-4 # これ小さすぎ？
        alpha = calc_alpha(i[0], e)
        L1_ratio = calc_L1_ratio(i[0], i[1], e)
        list2.append(alpha)
        list2.append(L1_ratio)
        ls.append(list2)

    regcoef_df = pd.DataFrame(ls)
    regcoef_df.to_csv("regcoef.csv",header=False, index=False, sep="\t")
    return ls

def calc_EN(x, y, coef):
    """
    エラスティックネットの計算をする。
    :param x: エラスティックネットの説明変数
    :param y: エラスティックネットの目的変数
    :param coef:エラスティックネットのハイパーパラメータ
    :return:エラスティックネットの計算結果。パレート集合。
    """
    data_x = pd.DataFrame(x)
    data_y = pd.DataFrame(y)
    ls = [] # TODO change to Pandas dataframe?
    for i in coef:
        # TODO suspicious epsilon
        if i[1] < 1 - 1e-4: # L1_ratio が大体 1e-4 より低い時にElastic Netが収束しない問題がある。　#todo小さい時は置き換える
            i[1] += 1e-4 # continue # 要実験
        elastic_net = ElasticNet(alpha = i[0] + 0.04, #0.04より小さいと収束しない　# TODO +0.04 をやめて線型回帰で置き換える
                                 l1_ratio = i[1])  #li_ratio alpha 足さないとエラーが出る。 /Users/zaizenkiichi/PycharmProjects/pythonProject2/venv/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.473e+00, tolerance: 5.000e-04 Linear regression models with null weight for the l1 regularization term are more efficiently fitted using one of the solvers implemented in sklearn.linear_model.Ridge/RidgeCV instead.model = cd_fast.enet_coordinate_descent(
        elastic_net = elastic_net.fit(data_x, data_y)
        elastic_net_np = elastic_net.coef_.round(3)
        elastic_net_list = elastic_net_np.tolist()
        ls.append(elastic_net_list)

    return ls

def f3(coef):
    X = np.array(coef)
    return np.linalg.norm(X, ord = 2)**2

# TODO suspicious epsilon
eps = 1e-4 # 16 で良いのか？ 16はかなり小さい（数値誤差に埋もれやすい） 1e-4, 1e-3, 1e-2 などで実験

# TODO what's this function?
def f1c(data_x, data_y, coef):
    #calc 1/2M||X0 - y||^2
    #np.matmul(data_x, coef.T) = X0
    #(np.matmul(data_x, coef.T) - data_y) = X0 - y
    coef = np.array(coef)
    M = 0
    for _ in data_x:
        M += 1
    g = (np.linalg.norm((np.matmul(data_x, coef.T) - data_y).T, ord = 2)**2)/(2*M)
    return g + eps * f3(coef)

# TODO what's this function?
def f2c(coef):
    #calc |0|
    X = np.array(coef)
    return np.linalg.norm(X, ord = 1) + eps * f3(coef)

# TODO what's this function?
def f3c(coef):
    #calc 1/2||0||^2
    X = np.array(coef)
    return (np.linalg.norm(X, ord = 2)**2)/2 + eps * f3(coef)

def calc_PF(x, y, pareto_set):
    """
    パレートフロントを計算する。
    :param x: エラスティックネットの説明変数
    :param y: エラスティックネットの目的変数
    :param pareto_set: パレート集合
    :return: パレートフロント
    """
    ls = []
    for i in pareto_set:
        list2 = []
        a = f1c(x, y, i)
        b = f2c(i)
        c = f3c(i)

        d = a.tolist()
        e = b.tolist()
        f = c.tolist()
        list2.append(d)
        list2.append(e)
        list2.append(f)
        ls.append(list2)

    df = pd.DataFrame(ls)
    df.to_csv("pareto_front.csv",header=False, index=False, sep="\t")
    return ls

def make_data_file():
    """
    パレート集合とパレートフロントを一つのファイルにまとめる。
    :return:まとめたファイル
    """
    pareto_set = np.loadtxt("elastic_net.csv")
    pareto_front = np.loadtxt("pareto_front.csv")
    ls = []
    pareto_set_list = pareto_set.tolist()
    pareto_front_list = pareto_front.tolist()
    for i in range(len(pareto_set_list)):
        list2 = []
        for j in range(len(pareto_set_list[0])):
            list2.append(pareto_set_list[i][j])
        for k in range(3):
            list2.append(pareto_front_list[i][k])
        ls.append(list2)
    df = pd.DataFrame(list)
    df.to_csv("dataf123.csv",header=False, index=False, sep="\t")
    return ls

def bezier_fit(
        triangle: [int] = [], # default: largest triangle
        loop: [int] = 5, #何回ループするのか
        degrees: [int] = list(range(1, 31)), # orders of polynomial in Bezier simplex fitting [1, 2,.., 30]
        datax: [float] = [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0], [7.0, 8.0, 9.0], [12.0, 11.0, 10.0]],
        datay: [float] = [1.0, 2.0, 3.0, 4.0]
    ) -> ([float], [float]):
    """
    ベジエ単体フィッティングを行う。１次から１５次までフィッティングする。
    :param triangle: フィッティングする三角形
    :param datax: 説明変数
    :param datay: 目的変数
    :return:テスト誤差のlist、計算時間のlist
    
    The default data are from Mizota et al. (arXiv:2106.12704v1). However, in their computation they normalize the data beforehand.
    As we are supplying the original datax and datay, the fitting result looks different. 
    """
    Llist_ave = [] #テスト誤差が入るリスト
    Llist_time = [] #計算時間が入るリスト

    w = make_w(triangle=triangle) #参照三角形を生成する関数
    #print(w)
    coef = trans_param(w) #入力の三角形に合わせてwをElastic Netのハイパーパラメタに変換
    #print(coef)
    pareto_set = calc_EN(datax, datay, coef)  #パレートセットを計算

    class temp: # Dirty trick: convert the list to pandas dataframe
        df_pareto_set = pd.DataFrame(pareto_set)

        # fig = px.scatter_3d(df_pareto_set, x=0, y=1, z=2, title="The input solution map (path) of elastic net")
        # fig.show()

        #TODO suspicous writing (this file is actually used for joining CSVS; should be done without IO, though)
        df_pareto_set.to_csv("elastic_net.csv",header=False, index=False, sep="\t")

    f = calc_PF(datax, datay, pareto_set) #パレートフロントを計算
    #make_data_file() #パレートセットのファイルとパレートフロントのファイルを合体
    sfo = []
    for i in range(len(pareto_set)):
        list2 = []
        for j in range(len(pareto_set[0])):
            list2.append(pareto_set[i][j])
        for k in range(3):
            list2.append(f[i][k])
        sfo.append(list2)

    N_DATA = len(w)
    N_TEST = N_DATA // 10
    N_TRAIN = N_DATA - N_TEST
    data_indices = list(range(N_DATA))  # [0, ..., 5150]
    test_indices = random.sample(data_indices, N_TEST)  # indices to test data
    train_indices = [i for i in data_indices if i not in test_indices]  # indices to training data

    t = maket(w, triangle=triangle)
    tt = []
    ss = [] # Pareto set
    ff = [] # Pareto front
    sf = [] # Pareto set x front

    for i in train_indices:
        tt.append(t[i])
        ss.append(pareto_set[i])
        ff.append(f[i])
        sfi = []
        for j in range(len(pareto_set[0])):
            sfi.append(pareto_set[i][j])
        for k in range(3):
            sfi.append(f[i][k])
        sf.append(sfi)
    tt = torch.tensor(tt)
    ss = torch.tensor(ss)
    ff = torch.tensor(ff)
    sf = torch.tensor(sf)
    t = torch.tensor(t)
    s = torch.tensor(pareto_set)
    f = torch.tensor(f)
    sfo = torch.tensor(sfo)
    #xdf = pd.DataFrame(s[:, 3:6], columns=['01','02','03'])
    #xdf = pd.DataFrame(f, columns=['f1','f2','f3'])
    xdf = pd.DataFrame(sfo[:, 0:3], columns=['sf1','sf2','sf3'])
    # fig = px.scatter_3d(xdf, x='sf1', y='sf2', z='sf3')
    # fig.show()

    # Use torch_bsf to learn the solution paths (i.e. solution map)

    for j in range(loop):#実験の回数
        list_ave = []
        list_time = []
        for d in degrees:#どの次数まで計算するか
            start = time.perf_counter()
            #b = torch_bsf.fit(params=tt, values=ss, degree=d)
            b = torch_bsf.fit(params=tt, values=sf, degree=d) # w -> fの対応関係を訓練したベジエ単体：単体から3次元空間への関数
            end = time.perf_counter()
            tm = end - start
            _, bts = b.meshgrid(num=100)
            bts = bts.detach()
            # df = pd.DataFrame(bts[:, 0:3],columns=['sf1','sf2','sf3'])
            # fig = px.scatter_3d(df, x='sf1', y='sf2', z='sf3')
            # fig.show()
            test_error = 0
            for i in test_indices:
                #test_error += np.square(s[i].detach().numpy() - b(t)[i].detach().numpy())
                test_error += np.square(sfo[i].detach().numpy() - b(t)[i].detach().numpy())
            test_error = np.mean(test_error) # 1つのパレートフロント全体/一部から1つのベジエ単体全体へのテスト誤差

            list_ave.append(test_error)
            list_time.append(tm)
        Llist_ave.append(list_ave)
        Llist_time.append(list_time)

    return (pd.DataFrame(Llist_ave),
            pd.DataFrame(Llist_time))


class Test_bezier (unittest.TestCase):
    def test_bezier(self):
    
        # QSAR fish data
        # names = ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP", "quantitative response, LC50 [-LOG(mol/L)]"]
        # df = pd.read_csv("resources/example-data/QSAR-fish/qsar_fish_toxicity.csv",
        #                  sep=";",
        #                  names=names
        #                  )
        # x = df[names[:-1]]
        # y = df[[names[-1]]]
        # x = x.values.tolist()
        # y = y.values.flatten().tolist()

        # Test without subdivision
        avedf, timedf = bezier_fit(triangle=[], loop=10, degrees=[0, 8],
                                   # datax=x, datay=y  # Load fish. (Comment out this line to do this fitting with the default toy data)
                                   )
        # avedf.to_csv("ave_err.csv",header=False, index=False, sep="\t")
        # timedf.to_csv("calc_time.csv",header=False, index=False, sep="\t")

        degree_0_error = np.median(avedf[0])
        degree_8_error = np.median(avedf[1])

        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_error / degree_0_error),
                log10(
                    0.01),# We get roughly 100x improvements in approximating the input surface
                delta=0.5
            )

        degree_0_time = np.median(timedf[0])
        degree_8_time  = np.median(timedf[1])
        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_time / degree_0_time),
                log10(
                    10.0), # We get roughly 10x speed up in timing, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware
                delta=0.5
            )

        avedf, timedf = bezier_fit(triangle=[Subdivision.triangle_center], loop=10, degrees=[0,8]
                                      # datax=x, datay=y  # Load fish. (Comment this line to do this fitting with the default toy data)
                                      )

        degree_0_error_triangle_center = np.median(avedf[0])
        degree_8_error_triangle_center = np.median(avedf[1])

        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_error_triangle_center/degree_0_error_triangle_center),
                log10(
                    0.003),# We get roughly 1/0.03 times improvements in approximating the input surface
                delta=0.5
            )
        
        degree_0_time_triangle_center = np.median(timedf[0])
        degree_8_time_triangle_center  = np.median(timedf[1])

        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_time_triangle_center/degree_0_time_triangle_center),
                log10(
                    12.0), # We get roughly 12x speed up in timing, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware
                delta=0.5
            )

        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_0_error_triangle_center / degree_0_error),
                log10(
                    0.1), # We get roughly this improvement in error, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware
                delta=0.5
            )
            
        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_0_time_triangle_center / degree_0_time),
                log10(
                    0.5), # We get roughly this improvement in time, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware,
                delta=0.5
            )

        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_error_triangle_center / degree_8_error),
                log10(
                    0.02), # We get roughly this improvement in error, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware,
                delta=0.5
            )
            
        with self.subTest():
            self.assertAlmostEqual(
                log10(
                    degree_8_time_triangle_center  / degree_8_time),
                log10(
                    0.5), # We get roughly this improvement in time, using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware,
                delta=0.5
            )
