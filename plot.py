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
    header, data = load_data('slac_triplet1.csv')
    
    steps = data[:,0]
    rs = data[:,1]
    # method = "mixed"

    # plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(steps, rs, '-o', markersize=1, label="Triplet 1")

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time step', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("SLAC Learning Curve", fontsize = 24)
    plt.savefig("plots/slac.png")