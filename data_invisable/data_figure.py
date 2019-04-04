# _*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt

Aupr_value = np.loadtxt("CMLDR_Aupr_values.txt").ravel()
plt.plot([i for i in range(1,11)],Aupr_value)
plt.axis([0,10,0.2,0.47])
plt.xlabel("times")
plt.ylabel("aupr_values")
plt.savefig("CMLDR_Aupr_values.png")
plt.show()

precision_data = np.loadtxt("CMLDR_precisions.txt")
for n_column in range(precision_data.shape[1]):
    precision = precision_data[:,n_column].ravel()
    plt.plot([i for i in range(1,11)],precision,label="k="+str(n_column+1))
plt.xlabel("times")
plt.ylabel("precisions")
plt.savefig("CMLDR_precisions.png")
plt.legend()
plt.show()

recall_data = np.loadtxt("CMLDR_recalls.txt")
for n_column in range(recall_data.shape[1]):
    precision = recall_data[:,n_column].ravel()
    plt.plot([i for i in range(1,11)],precision,label="k="+str(n_column+1))
plt.xlabel("times")
plt.ylabel("recalls")
plt.savefig("CMLDR_recalls.png")
plt.legend()
plt.show()
