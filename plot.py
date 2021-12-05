import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        data = np.array(list(reader)).astype(float)
    return header, data

if __name__ == '__main__':
    header, data = load_data('data/name-of-experiment/name-of-experiment_2021_11_19_14_23_55_0000--s-0/progress.csv')
    ns = range(data.shape[0])
    print(header)
    print(header[10])
    print(header[11])
    print(header[12])
    avs = data[:,11]
    maxs = data[:,10]
    mins = data[:,12]

    # method = "mixed"

    plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(ns, avs, '-o', markersize=1, label="Train")

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time step', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("SAC PCA", fontsize = 24)
    plt.savefig("plots/SAC PCA.png")