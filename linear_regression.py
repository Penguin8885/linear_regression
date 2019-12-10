import sys
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

####################

def f_true(x):
    return np.sin(10*(x+0.15)) / (10*(x+0.15)) # sinc
    # return 0.1*np.sin(20*x) + 10    # sin

def noise(dim, sigma2=3):
    return np.random.normal(0, np.sqrt(sigma2), dim)

####################

def Polynomial(x, size):    # 多項式回帰を行う場合の計画行列を返却する
    '''
    x = [1, 2, 3]
    return = [[1, 1, 1]
              [1, 4, 8]
              [1, 9, 27]]
    '''
    return np.vander(x, N=size, increasing=True) # ヴァンデルモンド行列

def RBF():                  # RBFで回帰を行う場合の計画行列を返却する
    pass

def Fourier():              # フーリエ級数で回帰を行う場合の計画行列を返却する
    pass

####################

def ML(y, A):
    # np.random.shuffle(A)
    if A.shape[0] < A.shape[1]:
        raise Exception('ERROR: cannnot calc ML, data size is too few')
    elif A.shape[0] == A.shape[1]:
        w = la.solve(A, y)          # (A^T*A)*w = A^T*y  ->  A*w = y
    else:
        w = la.solve(A.T@A, A.T@y)  # (A^T*A)*w = A^T*y
    return w

def MAP_Ridge(y, A, lambda_):
    I = np.eye(A.shape[1])
    w = la.solve(A.T@A + lambda_*I, A.T@y)  # (A^T*A + λI)*w = A^T*y
    return w

def MAP_Lasso(y, A, lambda_):
    pass

####################

def regression(w):
    pass

if __name__ == '__main__':
    np.random.seed(777)                     # 乱数シード固定
    x_range = [0.001, 1]                    # 表示範囲

    #### #### 真の曲線を作成・表示 #### ####
    x_true = np.linspace(x_range[0], x_range[1], 1000)
    y_true = f_true(x_true)
    # plt.plot(x_true, y_true, linestyle='--', label='f_true')

    #### #### サンプル点を取得・表示 #### ####
    sample_num = 20                         # サンプル数
    sigma2 = 0.004                          # サンプルに加わるノイズの分散
    x_sample_lower = x_range[0] + ((x_range[1] - x_range[0]) * 0.02)
    x_sample_upper = x_range[1] - ((x_range[1] - x_range[0]) * 0.02)
    x_sample = np.random.uniform(x_sample_lower, x_sample_upper, sample_num)
    y_sample = f_true(x_sample) + noise(len(x_sample), sigma2=sigma2)
    y_sample[12] = 0.2
    plt.scatter(x_sample, y_sample, c='red', marker='.', s=100, label='sample')

    #### #### 回帰 #### ####
    # 計画行列を計算
    A_size = 16                             # 計画行列Aのサイズを決定
    A = Polynomial(x_sample, size=A_size)   # x_sampleから多項式回帰で計画行列Aを構成
    print("Polynomial Regression (degree=%d)" % (A_size - 1))

    # 各種の重み係数を計算
    ## 最尤推定
    w_ML = ML(y_sample, A)                  # wの最尤推定解を求める
    print(w_ML)
    print(la.norm(w_ML))                    # 過学習のレベルを見るためにノルムを表示
    ## MAP推定(Ridge)
    w_MAP_Ridge = MAP_Ridge(y_sample, A, lambda_=1e-4) # wのMAP解(Ridge)を求める
    print(w_MAP_Ridge)
    print(la.norm(w_MAP_Ridge))             # 過学習のレベルを見るためにノルムを表示

    # 回帰を描画
    x_reg = np.linspace(x_range[0], x_range[1], 1000)
    A = Polynomial(x_reg, size=A_size)      # x_regから多項式回帰で計画行列Aを構成
    ## 最尤推定
    y_ML = A@w_ML                           # 予測関数(線形回帰)でyを取得
    # plt.plot(x_reg, y_ML, label='ML')
    ## MAP推定(Ridge)
    y_MAP_Ridge = A@w_MAP_Ridge             # 予測関数(線形回帰)でyを取得
    # plt.plot(x_reg, y_MAP_Ridge, label='MAP(Ridge)')

    plt.plot(x_true, y_true, linestyle='-', label='Regression')
    plt.plot(x_reg, y_ML, label='Regression')
    plt.plot(x_reg, y_MAP_Ridge, label='MAP(Ridge)')

    #### #### グラフの表示 #### ####
    plt.ylim(-0.4, 0.8)
    plt.legend(loc='upper right')
    plt.show()
