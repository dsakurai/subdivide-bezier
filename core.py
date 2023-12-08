from math import log10
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import time
import torch
import torch_bsf
import unittest
import pytest_check
import plotly.express as px

class Subdivision:
    triangle_0      = 0
    triangle_1      = 1
    triangle_2      = 2
    triangle_center = 3

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

def generate_ws_evenly(
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
    ls = []
    for w1 in range(resolution + 1):
        for w2 in range(resolution - w1 + 1):
            w3 = resolution - w1 - w2

            w = (w1/resolution,
                 w2/resolution,
                 w3/resolution,
            )
            
            if in_triangle(triangle=triangle, w=w):
                ls.append(w)
    return ls

def generate_ws_randomly(
        number: int = 100,
        triangle: [int] = [] # default: largest triangle
):

    def random_w():
        """
        :return: A hyperparameter w that 
        """

        x = np.random.rand()
        y = np.random.uniform(0, 1-x)
        z = 1.0 - (x + y)

        y = min(y, 1.0)
        y = max(y, 0.0)

        z = min(z, 1.0)
        z = max(z, 0.0)

        return (x, y, z)
    
    ls = []
    while len(ls) != number:
        w = random_w()
        if in_triangle(triangle=triangle, w=w):
            ls.append(w)
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

def localize_ws(
        w,
        triangle: [int] = [] # default: largest triangle
):
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
    # The original function expects a set of ws.
    # We get a single w and wrap it in a new list [w] so that we can use the original function.
    # As we get the output as a list, we un-wrap the element that corresponds to the input w.
    # (In fact, the output list has length 1.)
    # TODO It's cleaner to re-write the original function and use that one instead.
    return localize_ws([w], triangle=triangle)[0]

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
def f1_perturbed(data_x, data_y, thetas, eps):
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
def f2_perturbed(thetas, eps):
    #calc |0|
    X = np.array(thetas)
    return np.linalg.norm(X, ord = 1) + eps * f3(thetas)

# TODO what's this function?
def f3_perturbed(thetas, eps):
    #calc 1/2||0||^2
    X = np.array(thetas)
    return (np.linalg.norm(X, ord = 2)**2)/2 + eps * f3(thetas)

def f_perturbed(x, y, thetas, eps = 1e-4):
    """
    パレートフロントを計算する。
    :param x: エラスティックネットの説明変数
    :param y: エラスティックネットの目的変数
    :param eps: This epsilon perturbs f1, f2, f3 as specified by Mizota et al. It has to be rather large as this is used to perturb f. Otherwise, there's no point perturbing f
    :param pareto_set: パレート集合の点 (theta_0, theta_1, ...)
    :return: パレートフロント
    """
    return [
        f1_perturbed(x, y, thetas, eps=eps).tolist(),
        f2_perturbed(thetas, eps=eps).tolist(),
        f3_perturbed(thetas, eps=eps).tolist()
    ]


def split_test_train(num_points):
    num_tests = num_points // 10

    test_indices = np.random.randint(low=0, high=num_points - 1, size=num_tests).tolist()  # indices of test data
    train_indices = [i for i in range(num_points) if i not in test_indices]  # indices of training data

    return test_indices, train_indices

def thetas_and_fs(elastic_net_thetas, data_x, data_y):
    """Ccoordinate positions (i.e. list of Pareto set x Pareto front in elastic net)"""
    return elastic_net_thetas + f_perturbed(data_x, data_y, elastic_net_thetas)

def fit_bezier_simplex(ws_global, triangle, degree, elastic_net_solutions):

    # Local coordinates for this triangle
    w_local_train         = torch.tensor(localize_ws(ws_global, triangle=triangle))
    elastic_net_solutions = torch.tensor(elastic_net_solutions)
    
    # w -> fの対応関係を訓練したベジエ単体：単体から3次元空間への関数
    time_start = time.perf_counter()
    bezier_simplex = torch_bsf.fit(
        params=w_local_train,
        values=elastic_net_solutions,
        degree=degree)
    time_end = time.perf_counter()
    
    return bezier_simplex, (time_end - time_start)

def experiment_bezier(
        triangle: [int] = [],
        num_experiments: [int] = 5,
        degrees: [int] = list(range(1, 31)),
        data_x: [float] = [[ 1.0,  2.0,  3.0],
                          [ 6.0,  5.0,  4.0],
                          [ 7.0,  8.0,  9.0],
                          [12.0, 11.0, 10.0]],
        data_y: [float] = [1.0, 2.0, 3.0, 4.0],
        seed=0
    ) -> ([float], [float]):
    """
    Run experiments of Bezier fitting.
    :param triangle: The choice of the subdivided triangle. The default is actually the largest triangle, i.e. the whole triangle
    :param num_experiments:  Run the fitting and evaluation this many times
    :param degrees: orders of polynomial in Bezier simplex fitting. The default is [1, 2,.., 30].
    :param data_x: variables to be fed for Elastic Net regression
    :param data_y: observed values for the variables data_x
    :return: two lists: one for errors and another for duration
    
    The default data are from Mizota et al. (arXiv:2106.12704v1). However, in their computation they normalize the data beforehand.
    As we are supplying the original data_x and data_y, the fitting result looks different. 
    """

    # Fit the Bezier simplex to the solution paths (i.e. solution map)
    approximation_errors = [] #テスト誤差が入るリスト
    training_timings = [] #計算時間が入るリスト
    for j in range(num_experiments):#実験の回数
        approximation_errors_j = []
        training_timings_j = []

        for d in degrees:
        
            np.random.seed(seed)

            # Hyperparameters w in the triangle
            w_global = generate_ws_evenly(resolution=40, triangle=triangle) #参照三角形を生成する関数
            
            # Ground truth: the manifold to be approximated by the Bezier simplex.
            elastic_net_solutions = fit_elastic_nets(data_x, data_y, w_global)

            # Learn the solution space of elastic net
            bezier_simplex, duration = fit_bezier_simplex(
                ws_global=w_global,
                triangle=triangle,
                # Pick elastic net results for training
                elastic_net_solutions=[thetas_and_fs(elastic_net_solution, data_x, data_y) for elastic_net_solution in elastic_net_solutions],
                degree=d
                )
                
            # size of sampled hyperparameter set for testing
            test_size = len(w_global)//10
            
            w_global_test = generate_ws_randomly(number=test_size, triangle=triangle)
            
            elastic_net_thetas_test = fit_elastic_nets(data_x, data_y, w_global_test)
            
            errors = [
                np.square(
                    # [w1, w2, w3] for index i
                    thetas_and_fs(elastic_net_thetas_test[i], data_x, data_y)
                    - bezier_simplex([
                        localize_w(w_global_test[i], triangle=triangle)
                    ]).detach().numpy())
                for i in range(test_size)
            ]

            approximation_errors_j.append(
                np.mean(errors)) # 1つのパレートフロント全体/一部から1つのベジエ単体全体へのテスト誤差
            training_timings_j.append(duration)

        approximation_errors.append(approximation_errors_j)
        training_timings.append(training_timings_j)

    return (pd.DataFrame(approximation_errors, columns=degrees),
            pd.DataFrame(training_timings,     columns=degrees))


class MyTest(unittest.TestCase):

    def assertAlmostEqual(self, first, second, relative=0.1):
    
        allow = second * relative # tolerance relative to the expectation
        
        super().assertAlmostEqual(
            first=first,
            second=second,
            delta=allow
        )
    def test_bezier(self):

        # Test without subdivision
        approximation_errors, training_timings = experiment_bezier(triangle=[], num_experiments=5, degrees=[0, 2],
                                          # data_x=x, data_y=y  # Load fish. (Comment out this line to do this fitting with the default toy data)
                                          )
        
        degree_0_error = np.median(approximation_errors[0])
        degree_2_error = np.median(approximation_errors[2])
        
        self.assertAlmostEqual(
            degree_2_error/degree_0_error, 0.09, relative=0.5
        )
        
        degree_0_time = np.median(training_timings[0])
        degree_2_time = np.median(training_timings[2])

        with self.subTest():
            self.assertAlmostEqual(
                degree_2_time/degree_0_time,
                2.8,
                relative=0.5
            )

        approximation_errors, training_timings = experiment_bezier(triangle=[Subdivision.triangle_center], num_experiments=5, degrees=[0, 2]
                                          # data_x=x, data_y=y  # Load fish. (Comment this line to do this fitting with the default toy data)
                                          )
        
        degree_0_error_triangle_center = np.median(approximation_errors[0])
        degree_2_error_triangle_center = np.median(approximation_errors[2])

        self.assertAlmostEqual(
            degree_2_error_triangle_center/degree_0_error_triangle_center,
            0.04,
            relative=0.5
        )
        
        degree_0_time_triangle_center = np.median(training_timings[0])
        degree_2_time_triangle_center = np.median(training_timings[2])

        with self.subTest():
            self.assertAlmostEqual(
                degree_2_time_triangle_center/degree_0_time_triangle_center,
                3.1,
                relative=0.5
            )

        with self.subTest():
            self.assertAlmostEqual(
                degree_0_error_triangle_center/degree_0_error,
                0.4,
                relative=0.5
            )

        with self.subTest():
            self.assertAlmostEqual(
                degree_0_time_triangle_center/degree_0_time,
                1.0,
                relative=0.5
            )

        self.assertAlmostEqual(
            degree_2_error_triangle_center/degree_2_error,
            0.16,
            relative=0.5
        )

        with self.subTest():
            self.assertAlmostEqual(
                degree_2_time_triangle_center/degree_2_time,
                0.95,
                relative=0.5
            )
