# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import time
import random
import plotly.express as px

class Subdivision:
    triangle_0      = 0
    triangle_1      = 1
    triangle_2      = 2
    triangle_center = 3

eps = 0.0000000001

# Number of flips in the triangle
def flipped(upper_triangle):
    num_flips = upper_triangle.count(Subdivision.triangle_center)
    return num_flips % 2 == 1

def in_triangle_(smallest_triangle: [int], w: [float], c= 0):
    """
    Check if the point `w` is in the specified `triangle`.
    Presumption: in_triangle() for higher level all returns True.
    :param smallest_triangle: hierarchy of triangles
    :param w: barycentric coordinates of the point.
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

def make_t(
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

    t = triangle[-1]
    if t in [Subdivision.triangle_0, Subdivision.triangle_1, Subdivision.triangle_2]:
        if Subdivision.triangle_0 in triangle:
            i = -1
            while True:
                if triangle[i] == Subdivision.triangle_0:
                    if i != -1:
                        border[0] = boundary(triangle=triangle[:i+1])
                        break
                    else:
                        border[0] = boundary(triangle=triangle)
                        break
                i -= 1

        if Subdivision.triangle_1 in triangle:
            j = -1
            while True:
                if triangle[j] == Subdivision.triangle_1:
                    if j != -1:
                        border[1] = boundary(triangle=triangle[:j+1])
                        break
                    else:
                        border[1] = boundary(triangle=triangle)
                        break
                j -= 1

        if Subdivision.triangle_2 in triangle:
            k = -1
            while True:
                if triangle[k] == Subdivision.triangle_2:
                    if k != -1:
                        border[2] = boundary(triangle=triangle[:k+1])
                        break
                    else:
                        border[2] = boundary(triangle=triangle)
                        break
                k -= 1
    else:
        border[0] = boundary(triangle=triangle[:-1]+[Subdivision.triangle_0])
        border[1] = boundary(triangle=triangle[:-1]+[Subdivision.triangle_1])
        border[2] = boundary(triangle=triangle[:-1]+[Subdivision.triangle_2])

    print(border)
    tlist = trans(triangle, border, w)
    # df = pd.DataFrame(tlist)
    return tlist

def maket(
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
    print(border)
    tlist = trans(triangle, border, w)
    # df = pd.DataFrame(tlist)
    return tlist



start_time = time.perf_counter()
#testriangle = []
testriangle = [2]
#testriangle = [0]
#testriangle = [3]
#testriangle = [0,0]
#testriangle = [0,1]
#testriangle = [0,3]
#testriangle = [3,0]
#testriangle = [3,3]

w = make_w(triangle=testriangle)
wdf = pd.DataFrame(w,columns=['w1','w2','w3'])
fig = px.scatter_3d(wdf, x='w1', y='w2', z='w3')
fig.show()

t = maket(triangle=testriangle)
tdf = pd.DataFrame(t,columns=['w1','w2','w3'])
fig = px.scatter_3d(tdf, x='w1', y='w2', z='w3')
fig.show()

end_time = time.perf_counter()
print(len(w))
print(len(t))
print(end_time - start_time)
