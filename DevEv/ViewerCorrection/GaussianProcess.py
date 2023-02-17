import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scipy.signal import find_peaks
from scipy.stats import multivariate_normal

def read_data(filename):
    with open(filename, "r") as f:
        data = f.readlines()

    x_list = []
    count = 0
    for d in data:
        if len(d) <= 10: continue
        fid, x, y, z, y, p, r, _,_,_ = d.split(",")
        x_list.append([float(x), float(y), float(z), float(y), float(p), float(r)])
        count += 1
        #if count > 2100: break
    x_list = np.array(x_list)
    x_list =  x_list[1:] - x_list[:-1]
    y_list = np.linspace(0, 1, len(x_list))

    return x_list, y_list


def GP(x_tr, tau = 10):
    mean, var, value = [], [], []
    N, D = x_tr.shape
    for t, x in enumerate(x_tr):
        tau_min = max(0,t-tau)
        tau_max = min(N,t+tau+1)

        segment = x_tr[tau_min:tau_max]
        mu = np.mean(segment, axis = 0)
        if np.nan in mu or np.inf in mu:
            print(t, tau_min, tau_max, segment.shape)
            exit()
        temp = segment - mu
        sigma = np.dot(temp.T, temp)/(tau+1)

        mvg = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
        v = mvg.logpdf(x)/ mu.shape[0]


        var.append(sigma)
        mean.append(mu)
        value.append(v)

    mean = np.array(mean)
    var = np.array(var)
    value = np.array(value)

    return mean, var, value

def get_uncertainty(x_tr, max_n=None):
    mean, var, value = GP(x_tr)
    peaks, _ = find_peaks(value, height=2.0, distance=40, prominence=0.05)
    selected = value[peaks]
    if max_n is not None:
        ind_selected = value[peaks].argsort()[::-1]
        ind_selected = ind_selected[:max_n]
        selected = selected[ind_selected]
        peaks = peaks[ind_selected]
    return peaks, selected

def uncertainty(filename):

    x_tr, y_tr = read_data(filename)
    mean, var, value = GP(x_tr)

    peaks, _ = find_peaks(value, height=2.0, distance=40, prominence=0.05)

    m = mean[:,0]
    err = np.sqrt(var[:, 0, 0])
    plt.plot(np.arange(len(m)),  m+err, "--r", label='std', linewidth=1)
    plt.plot(np.arange(len(m)),  m-err, "--r", linewidth=1)
    plt.plot(np.arange(len(m)),  m, "-b", label='Mean')
    plt.title("GP")
    plt.ylabel("x Offset")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("data/GP_x.png")
    plt.close()

    plt.plot(np.arange(len(value)),  value, "-g", label='logpdf')
    plt.plot(peaks, value[peaks], "xk", label="selection")
    plt.title("Frame selection")
    plt.ylabel("logpdf")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("data/GP_selected.png")
    plt.close()


    peaks = [str(p) for p in peaks]
    with open("data/results.txt","a") as f:
        f.write(",".join(peaks))
    return peaks

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--num-samples", default=15, type=int)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("-f", default="data/attention.txt", type=str)
    args = parser.parse_args()

    p = uncertainty(args.f)
    
