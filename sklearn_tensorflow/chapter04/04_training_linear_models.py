# _*_ coding: utf-8 _*_
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'training_linear_models'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,'images',CHAPTER_ID)
os.makedirs(IMAGES_PATH,exist_ok=True)

def save_fig(fig_id,tight_layout=True,fig_extension='png',resolution=300):
    path = os.path.join(IMAGES_PATH,fig_id+'.'+fig_extension)
    print('Saving figure',fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format=fig_extension,dpi=resolution)

# ignore useless warning
import warnings
warnings.filterwarnings(action='ignore',message='internal gelsd')

# np.random.seed(42)
X = 2 * np.random.randn(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

# plt.plot(X,y,'b.')
# plt.xlabel('$x_1$',fontsize=18)
# plt.ylabel('$y$',rotation=0,fontsize=18)
# plt.axis([0,2,0,15])
# plt.show()

X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_best)

# plt.plot(X_new,y_predict,'r-',linewidth=2,label='Precision')
# plt.plot(X,y,'b.')
# plt.xlabel('$x_1$',fontsize=18)
# plt.ylabel('$y$',rotation=0,fontsize=14)
# plt.axis([0,2,0,15])
# plt.show()

# eta = 0.1
# n_iterations = 1000
# m = 100
#
# theta = np.random.randn(2,1)
#
# for iteration in range(n_iterations):
#     gradient = 2 / m * X_b.T.dot(X_b.dot(theta)-y)
#     theta = theta - eta * gradient

theta_path_bad = []
def plot_gradient_descent(theta,eta,theta_path=None):
    m = len(X_b)
    plt.plot(X,y,'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else 'r--'
            plt.plot(X_new,y_predict,style)
        gradients =  2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('$x_1$',fontsize=18)
    plt.axis([0,2,0,15])
    plt.title(r'$\eta = {}$'.format(eta),fontsize=16)

# np.random.seed(42)
theta = np.random.randn(2,1)

# plt.figure(figsize=(10,4))
# plt.subplot(131)
# plot_gradient_descent(theta,eta=0.02)
# plt.ylabel('$y$',fontsize=18)
# plt.subplot(132)
# plot_gradient_descent(theta,eta=0.1,theta_path=theta_path_bad)
# plt.subplot(133)
# plot_gradient_descent(theta,eta=0.3)
# plt.show()

theta_path_sgd = []
m = len(X_b)
# # np.random.seed(42)
#
n_epochs = 50
t0,t1 = 5,50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"  # not shown
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_bad.append(theta)
#
# plt.plot(X, y, "b.")                                 # not shown
# plt.xlabel("$x_1$", fontsize=18)                     # not shown
# plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
# plt.axis([0, 2, 0, 15])                              # not shown                                # not shown
# plt.show()
#

theta_path_mgd = []
n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0,m,minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bad)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)
plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bad[:, 0], theta_path_bad[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
save_fig("gradient_descent_paths_plot")
plt.show()