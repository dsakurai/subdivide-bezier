# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from math import log10
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
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

eps = 1e-4 # This has to be rather large as this is used to perturb f. Otherwise, there's no point perturbing f.

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

    if abs(sum(w) - 1.0 ) > 1e-4: raise Exception("Sum of w coordinates is not 1.")

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

def localize_w(
        w,
        triangle: [int] = [] # default: largest triangle
):
    """
    参照三角形をベジエ単体近似に使えるようにパラメータを変換する。変換する三角形の三辺の境界を見つけてtrans()を呼ぶ
    :param w: (w1, w2, w3) in the original triangle
    :param triangle: a triangle in the subdivision
    :return: w in coordinates localized within `triangle`
    """
    border = [0.0, 0.0, 0.0]
    if len(triangle) == 0:
        tlist = trans(triangle, border, w)
        # df = pd.DataFrame(tlist)
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
    tlist = trans(triangle, border, w)
    # df = pd.DataFrame(tlist)
    return tlist

def calc_alpha(w0, eps):
    w0 = max(w0,0.01) # TODO This 0.01 avoids explosion of alpha, but is 0.01 a good choice?
    return (1 - w0 + eps) / w0

def calc_L1_ratio(w0, w1, eps):
    b = w1 / (1 - w0 + eps)
    return b

def w_2_alpha_l1(w):
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

    # regcoef_df = pd.DataFrame(ls)
    return ls

def fit_elastic_nets(x, y, w):
    """
    エラスティックネットの計算をする。
    :param x: エラスティックネットの説明変数
    :param y: エラスティックネットの目的変数
    :param w:エラスティックネットのハイパーパラメータ
    :return:エラスティックネットの計算結果。パレート集合。
    """
    data_x = pd.DataFrame(x)
    data_y = pd.DataFrame(y)
    ls = [] # TODO change to Pandas dataframe?
    
    a_l = w_2_alpha_l1(w) # transform (w1, w2, w3) to the standard hyperparamenters, i.e. alpha and l1-ratio
    
    for alpha, l1_ratio in a_l:
        
        def fit(model):
            elastic_net      = model.fit(data_x, data_y)
            elastic_net_np   = elastic_net.coef_
            return elastic_net_np.tolist()

        # alpha and l1_ratio are prone to error. /Users/zaizenkiichi/PycharmProjects/pythonProject2/venv/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.473e+00, tolerance: 5.000e-04 Linear regression models with null weight for the l1 regularization term are more efficiently fitted using one of the solvers implemented in sklearn.linear_model.Ridge/RidgeCV instead.model = cd_fast.enet_coordinate_descent(
        
        if alpha < 0.04: # Scikit-learn tends to fail to converge ElasticNet under this setting.
            coefficients = fit(LinearRegression())
            coefficients = coefficients[0] # ElasticNet weirdness in Scikit-learn...
            
        elif l1_ratio < 1e-4: # L1_ratio が大体 1e-4 より低い時にElastic Netが収束しない問題がある。　#todo小さい時は置き換える
            coefficients = fit(Ridge(alpha=alpha))
            coefficients = coefficients[0] # Ridge regression weirdness in Scikit-learn...

        else: # Do ElasticNet
            coefficients = fit(ElasticNet(alpha=alpha, l1_ratio=l1_ratio))

        ls.append(coefficients)

    return ls

def f3(coef):
    X = np.array(coef)
    return np.linalg.norm(X, ord = 2)**2

# TODO what's this function?
def f1_perturbed(data_x, data_y, thetas):
    #calc 1/2M||X0 - y||^2
    #np.matmul(data_x, coef.T) = X0
    #(np.matmul(data_x, coef.T) - data_y) = X0 - y
    thetas = np.array(thetas)
    M = 0
    for _ in data_x:
        M += 1
    g = (np.linalg.norm((np.matmul(data_x, thetas.T) - data_y).T, ord = 2) ** 2) / (2 * M)
    return g + eps * f3(thetas)

# TODO what's this function?
def f2_perturbed(thetas):
    #calc |0|
    X = np.array(thetas)
    return np.linalg.norm(X, ord = 1) + eps * f3(thetas)

# TODO what's this function?
def f3_perturbed(thetas):
    #calc 1/2||0||^2
    X = np.array(thetas)
    return (np.linalg.norm(X, ord = 2)**2)/2 + eps * f3(thetas)

def f_perturbed(x, y, thetas):
    """
    パレートフロントを計算する。
    :param x: エラスティックネットの説明変数
    :param y: エラスティックネットの目的変数
    :param pareto_set: パレート集合の点 (theta_0, theta_1, ...)
    :return: パレートフロント
    """
    return [
        f1_perturbed(x, y, thetas).tolist(),
        f2_perturbed(thetas).tolist(),
        f3_perturbed(thetas).tolist()
    ]

def fit_bezier_simplex(
        df_pareto: pd.DataFrame,
        triangle: [int],
        params ,
        values ,
        degree: [int],
    ):
    """
    :param df_pareto: Pareto set, Pareto front, or both.
    :param triangle: The triangle in the subdivision.
    :param degree: Degree of the bezier simplex fitting
    :return: 
    """
    pass

def experiment_bezier(
        triangle: [int] = [],
        num_experiments: [int] = 5,
        degrees: [int] = list(range(1, 31)),
        datax: [float] = [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0], [7.0, 8.0, 9.0], [12.0, 11.0, 10.0]],
        datay: [float] = [1.0, 2.0, 3.0, 4.0]
    ) -> ([float], [float]):
    """
    Run experiments of Bezier fitting.
    :param triangle: The choice of the subdivided triangle. The default is actually the largest triangle, i.e. the whole triangle
    :param num_experiments:  Run the fitting and evaluation this many times
    :param degrees: orders of polynomial in Bezier simplex fitting. The default is [1, 2,.., 30].
    :param datax: variables to be fed for Elastic Net regression
    :param datay: observed values for the variables datax
    :return: two lists: one for errors and another for duration
    
    The default data are from Mizota et al. (arXiv:2106.12704v1). However, in their computation they normalize the data beforehand.
    As we are supplying the original datax and datay, the fitting result looks different. 
    """
    Llist_ave = [] #テスト誤差が入るリスト
    Llist_time = [] #計算時間が入るリスト
    
    np.random.seed(0)

    w = make_w(resolution=40, triangle=triangle) #参照三角形を生成する関数
    
    pareto_set = fit_elastic_nets(datax, datay, w)  #パレートセットを計算
    assert(len(w) == len(pareto_set))

    # fig = px.scatter_3d(pd.DataFrame(pareto_set), x=0, y=1, z=2, title="The input solution map (path) of elastic net")
    # fig.show()

    N_DATA = len(w)
    N_TEST = N_DATA // 10
    
        
    data_indices = list(range(N_DATA))  # [0, ..., 5150]
    test_indices = np.random.randint(low=0, high=N_DATA-1, size=N_TEST).tolist()  # indices of test data
    train_indices = [i for i in data_indices if i not in test_indices]  # indices of training data

    # Pareto set x Pareto front
    #
    # Ground truth coordinate positions (i.e. list of Pareto set x Pareto front in elastic net)
    pareto_set_x_front_ground_truth = torch.tensor([
        # Join (theta_0, theta_1,  ...) and (f_0, f_2, f_3)
        thetas + f_perturbed(datax, datay, thetas) for thetas in pareto_set
    ])
    #
    # Training dataset (subset of the ground truth)
    pareto_set_x_front_train = torch.tensor([
        pareto_set_x_front_ground_truth.detach().numpy()[i] for i in train_indices
    ])
    
    w_local = localize_w(w, triangle=triangle)
    
    w_local_train  = torch.tensor([w_local[id] for id in train_indices])
    w_local_tensor = torch.tensor( w_local    ) # TODO no point storing this in a tensor instance
    
    # xdf = pd.DataFrame(torch.tensor(pareto_set_x_front_ground_truth)[:, 0:3], columns=['sf1','sf2','sf3'])
    # fig = px.scatter_3d(xdf, x='sf1', y='sf2', z='sf3')
    # fig.show()

    # Use torch_bsf to learn the solution paths (i.e. solution map)

    for j in range(num_experiments):#実験の回数
        list_ave = []
        list_time = []

        for d in degrees:
        
            time_start = time.perf_counter()
            #
            bezier_simplex = torch_bsf.fit(
                params=w_local_train,
                values=pareto_set_x_front_train,
                degree=d) # w -> fの対応関係を訓練したベジエ単体：単体から3次元空間への関数
            #
            time_end = time.perf_counter()
            
            # _, bts = bezier_simplex.meshgrid(num=100)
            # bts = bts.detach() # TODO can we remove this line?
            # df = pd.DataFrame(bts[:, 0:3],columns=['sf1','sf2','sf3'])
            # fig = px.scatter_3d(df, x='sf1', y='sf2', z='sf3')
            # fig.show()
            errors = [
                np.square(
                    pareto_set_x_front_ground_truth[i].detach().numpy()
                    # [w1, w2, w3] for index i
                    - bezier_simplex(w_local_tensor.detach().numpy()[i].reshape(1,3)).detach().numpy())
                for i in test_indices
            ]
            test_error = np.mean(errors) # 1つのパレートフロント全体/一部から1つのベジエ単体全体へのテスト誤差

            list_ave.append(test_error)
            list_time.append(time_end - time_start)
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
        avedf, timedf = experiment_bezier(triangle=[], num_experiments=10, degrees=[0, 8],
                                          # datax=x, datay=y  # Load fish. (Comment out this line to do this fitting with the default toy data)
                                          )

        degree_0_error = np.median(avedf[0])
        degree_8_error = np.median(avedf[1])
        
        def assertAlmostEqual(value, expected):
            self.assertAlmostEqual(
                log10(
                    value),
                log10(
                    expected),# using the 32 core GPU on macOS Apple Sillicon M1 Max, but this is dependent on hardware
                delta=0.5,
                msg=f"actual value: {value}"
            )
            

        assertAlmostEqual(degree_8_error/degree_0_error, 0.01)

        degree_0_time = np.median(timedf[0])
        degree_8_time  = np.median(timedf[1])
        
        assertAlmostEqual(degree_8_time/degree_0_time, 10.0)

        avedf, timedf = experiment_bezier(triangle=[Subdivision.triangle_center], num_experiments=10, degrees=[0, 8]
                                          # datax=x, datay=y  # Load fish. (Comment this line to do this fitting with the default toy data)
                                          )

        degree_0_error_triangle_center = np.median(avedf[0])
        degree_8_error_triangle_center = np.median(avedf[1])

        assertAlmostEqual(degree_8_error_triangle_center/degree_0_error_triangle_center, 0.01)
        
        degree_0_time_triangle_center = np.median(timedf[0])
        degree_8_time_triangle_center  = np.median(timedf[1])

        assertAlmostEqual(degree_8_time_triangle_center/degree_0_time_triangle_center,13.0)

        assertAlmostEqual(degree_0_error_triangle_center/degree_0_error,0.4)
            
        assertAlmostEqual(degree_0_time_triangle_center/degree_0_time,0.5)

        assertAlmostEqual(degree_8_error_triangle_center/degree_8_error,0.2)
            
        assertAlmostEqual(degree_8_time_triangle_center/degree_8_time,0.5)
