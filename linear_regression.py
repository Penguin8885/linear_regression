import sys
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

#### システム環境の構成 ###########################################################

def func1(x):
    return np.sin(10*(x+0.15)) / (10*(x+0.15))  # sinc

def func2(x):
    return 0.1*np.sin(20*x) + 10                # sin

def noise(dim, sigma2=3):
    return np.random.normal(0, np.sqrt(sigma2), dim)

#### 計画行列を計算する関数 #######################################################

def Polynomial(x, size):    # 多項式回帰を行う場合の計画行列を返却する
    '''
    x = [1, 2, 3]
    return = [[1, 1, 1]
              [1, 4, 8]
              [1, 9, 27]]
    '''
    return np.vander(x, N=size, increasing=True) # ヴァンデルモンド行列

def RBF(x, size):           # RBFで回帰を行う場合の計画行列を返却する
    pass

def Fourier(x, size):       # フーリエ級数で回帰を行う場合の計画行列を返却する
    pass

#### 回帰問題のパラメータを計算する関数 ###########################################

def ML(y, A):
    # np.random.shuffle(A)
    if A.shape[0] < A.shape[1]:
        raise Exception('ERROR: cannnot calc ML, data size is too few')
    elif A.shape[0] == A.shape[1]:
        w = la.solve(A, y)                  # (A^T*A)*w = A^T*y  ->  A*w = y
    else:
        w = la.solve(A.T@A, A.T@y)          # (A^T*A)*w = A^T*y
    return w

def MAP_Ridge(y, A, lambda_):
    I = np.eye(A.shape[1])
    w = la.solve(A.T@A + lambda_*I, A.T@y)  # (A^T*A + λI)*w = A^T*y
    return w

def MAP_Lasso(y, A, lambda_):
    pass

#### 回帰問題の表示形 #############################################################

# 真の関数を表示
def plot_f_true(f_true, x_range):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f_true(x)
    plt.plot(x, y, linestyle='--', label='f_true')

# 真の関数からサンプリングをする関数
def sampling(num, f_true, sigma2, x_range, is_plt=True):
    # サンプル点の範囲を決定
    x_sample_lower = x_range[0] + ((x_range[1] - x_range[0]) * 0.02)
    x_sample_upper = x_range[1] - ((x_range[1] - x_range[0]) * 0.02)

    # サンプリング
    x_sample = np.random.uniform(x_sample_lower, x_sample_upper, num)
    y_sample = f_true(x_sample) + noise(num, sigma2=sigma2)
    y_sample[12] = 0.2 # !!!! 恣意的にデータ変更 !!!!

    # 表示
    if is_plt:
       plt.scatter(x_sample, y_sample, c='red', marker='.', s=100, label='sample')

    return x_sample, y_sample

# 回帰曲線を描く関数
def plot_linear_regression(x_sample, y_sample, design_mat_func, w_solver, x_range, label):
    # サンプルからwを解く
    A_sample = design_mat_func(x_sample)                # 計画行列を計算
    w = w_solver(y_sample, A_sample)                    # wを計算

    # wの表示
    print('label:')                                     # wの計算結果を表示
    print(w)
    print('norm(w): %f' % la.norm(w))                   # wのノルムを表示

    # wを用いて回帰曲線を描く
    x_reg = np.linspace(x_range[0], x_range[1], 1000)   # 曲線を引く範囲を決定
    A_reg = design_mat_func(x_reg)                      # 計画行列を計算
    y_reg = A_reg @ w                                   # 求めたwからyを計算
    plt.plot(x_reg, y_reg, label=label)                 # 回帰曲線の表示

################################################################################

if __name__ == '__main__':
    np.random.seed(777)                     # 乱数シード固定
    x_range = [0.001, 1]                    # 表示範囲

    #### #### 真のグラフの設定とサンプリング #### ####
    f_true = func1
    plot_f_true(f_true, x_range)
    x_samp, y_samp = sampling(20, f_true, 0.004, x_range, is_plt=True)

    #### #### ソルバーの設定 #### ####
    design_mat_func = lambda x : Polynomial(x, size=16)                 # 計画行列を多項式回帰で構成
    w_solver_ML         = lambda y, A : ML(y, A)
    w_solver_MAP_Ridge1 = lambda y, A : MAP_Ridge(y, A, lambda_=1e-9)
    w_solver_MAP_Ridge2 = lambda y, A : MAP_Ridge(y, A, lambda_=1e-6)
    w_solver_MAP_Ridge3 = lambda y, A : MAP_Ridge(y, A, lambda_=1e-3)
    # w_solver_MAP_Lasso  = lambda y, A : MAP_Lasso(y, A, lambda_=1e-6)

    #### #### 回帰曲線の表示 #### ####
    plot_linear_regression(x_samp, y_samp, design_mat_func, w_solver_ML, x_range, label='ML')
    plot_linear_regression(x_samp, y_samp, design_mat_func, w_solver_MAP_Ridge1, x_range, label='MAP(Ridge1)')
    plot_linear_regression(x_samp, y_samp, design_mat_func, w_solver_MAP_Ridge2, x_range, label='MAP(Ridge2)')
    plot_linear_regression(x_samp, y_samp, design_mat_func, w_solver_MAP_Ridge3, x_range, label='MAP(Ridge3)')
    # plot_linear_regression(x_samp, y_samp, design_mat_func, w_solver_MAP_Lasso, x_range, label='MAP(Lasso)')

    #### #### グラフの表示 #### ####
    plt.ylim(-0.4, 0.8)
    plt.legend(loc='upper right')
    plt.show()
