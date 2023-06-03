import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(e, n, R):
    return R * e ** n


def fit_n_Al(E, sy, s033, s066, s1):
    e = sy / E
    e033 = e + 0.033
    e066 = e + 0.066
    e1 = e + 0.1
    print("ε: ", e, "ε0.033: ", e033, "ε0.066: ", e066, "ε0.1: ", e1)
    return curve_fit(func, [e, e033, e066, e1], [sy, s033, s066, s1])


def fit_n_Ti(E, sy, s008, s015, s033):
    e = sy / E
    e008 = e + 0.008
    e015 = e + 0.015
    e033 = e + 0.033
    print("ε: ", e, "ε0.008: ", e008, "ε0.015: ", e015, "ε0.033: ", e033)
    return curve_fit(func, [e, e008, e015, e033], [sy, s008, s015, s033])


def main():
    ''' For Ti alloys
    sy = np.loadtxt("sigma_y.dat")
    s1 = np.loadtxt("sigma_0.008.dat")
    s2 = np.loadtxt("sigma_0.015.dat")
    s3 = np.loadtxt("sigma_0.033.dat")
    print("sy μ+σ: ", np.mean(sy), np.std(sy))
    print("s0.8% μ+σ: ", np.mean(s1), np.std(s1))
    print("s1.5% μ+σ: ", np.mean(s2), np.std(s2))
    print("s3.3% μ+σ: ", np.mean(s3), np.std(s3))
    '''
    ''' For Al alloys'''
    sy = np.loadtxt("sigma_y.dat")
    s1 = np.loadtxt("sigma_0.033.dat")
    s2 = np.loadtxt("sigma_0.066.dat")
    s3 = np.loadtxt("sigma_0.1.dat")
    print("sy μ+σ: ", np.mean(sy), np.std(sy))
    print("s3.3% μ+σ: ", np.mean(s1), np.std(s1))
    print("s6.6% μ+σ: ", np.mean(s2), np.std(s2))
    print("s10% μ+σ: ", np.mean(s3), np.std(s3))
    
    sy = np.mean(sy)
    s1 = np.mean(s1)
    s2 = np.mean(s2)
    s3 = np.mean(s3)

# Al6061
#    E = 66.8
#    (n, R), pcov = fit_n_Al(E, sy, s1, s2, s3)
# Al7075
    E = 70.1
    (n, R), pcov = fit_n_Al(E, sy, s1, s2, s3)
# Ti alloys
#    E = 110
#    (n, R), pcov = fit_n_Ti(E, sy, s1, s2, s3)
    print(n, pcov[0, 0] ** 0.5)
    print(R)


if __name__ == "__main__":
    main()
