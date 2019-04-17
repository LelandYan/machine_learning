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
# X = 2 * np.random.randn(100,1)
# y = 4 + 3 * X + np.random.randn(100,1)

# plt.plot(X,y,'b.')
# plt.xlabel('$x_1$',fontsize=18)
# plt.ylabel('$y$',rotation=0,fontsize=18)
# plt.axis([0,2,0,15])
# plt.show()

# X_b = np.c_[np.ones((100,1)),X]
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#
# X_new = np.array([[0],[2]])
# X_new_b = np.c_[np.ones((2,1)),X_new]
# y_predict = X_new_b.dot(theta_best)
#
# # plt.plot(X_new,y_predict,'r-',linewidth=2,label='Precision')
# # plt.plot(X,y,'b.')
# # plt.xlabel('$x_1$',fontsize=18)
# # plt.ylabel('$y$',rotation=0,fontsize=14)
# # plt.axis([0,2,0,15])
# # plt.show()
#
# # eta = 0.1
# # n_iterations = 1000
# # m = 100
# #
# # theta = np.random.randn(2,1)
# #
# # for iteration in range(n_iterations):
# #     gradient = 2 / m * X_b.T.dot(X_b.dot(theta)-y)
# #     theta = theta - eta * gradient
#
# theta_path_bad = []
# def plot_gradient_descent(theta,eta,theta_path=None):
#     m = len(X_b)
#     plt.plot(X,y,'b.')
#     n_iterations = 1000
#     for iteration in range(n_iterations):
#         if iteration < 10:
#             y_predict = X_new_b.dot(theta)
#             style = "b-" if iteration > 0 else 'r--'
#             plt.plot(X_new,y_predict,style)
#         gradients =  2 / m * X_b.T.dot(X_b.dot(theta) - y)
#         theta = theta - eta * gradients
#         if theta_path is not None:
#             theta_path.append(theta)
#     plt.xlabel('$x_1$',fontsize=18)
#     plt.axis([0,2,0,15])
#     plt.title(r'$\eta = {}$'.format(eta),fontsize=16)
#
# # np.random.seed(42)
# theta = np.random.randn(2,1)
#
# # plt.figure(figsize=(10,4))
# # plt.subplot(131)
# # plot_gradient_descent(theta,eta=0.02)
# # plt.ylabel('$y$',fontsize=18)
# # plt.subplot(132)
# # plot_gradient_descent(theta,eta=0.1,theta_path=theta_path_bad)
# # plt.subplot(133)
# # plot_gradient_descent(theta,eta=0.3)
# # plt.show()
#
# theta_path_sgd = []
# m = len(X_b)
# # # np.random.seed(42)
# #
# n_epochs = 50
# t0,t1 = 5,50
#
# def learning_schedule(t):
#     return t0 / (t + t1)
#
# theta = np.random.randn(2,1)
#
# for epoch in range(n_epochs):
#     for i in range(m):
#         if epoch == 0 and i < 20:
#             y_predict = X_new_b.dot(theta)
#             style = "b-" if i > 0 else "r--"  # not shown
#             plt.plot(X_new, y_predict, style)
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index+1]
#         yi = y[random_index:random_index+1]
#         gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients
#         theta_path_bad.append(theta)
# #
# # plt.plot(X, y, "b.")                                 # not shown
# # plt.xlabel("$x_1$", fontsize=18)                     # not shown
# # plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
# # plt.axis([0, 2, 0, 15])                              # not shown                                # not shown
# # plt.show()
# #
#
# theta_path_mgd = []
# n_iterations = 50
# minibatch_size = 20
#
# np.random.seed(42)
# theta = np.random.randn(2,1)
#
# t0, t1 = 200, 1000
# def learning_schedule(t):
#     return t0 / (t + t1)
#
# t = 0
# for epoch in range(n_iterations):
#     shuffled_indices = np.random.permutation(m)
#     X_b_shuffled = X_b[shuffled_indices]
#     y_shuffled = y[shuffled_indices]
#     for i in range(0,m,minibatch_size):
#         t += 1
#         xi = X_b_shuffled[i:i+minibatch_size]
#         yi = y_shuffled[i:i+minibatch_size]
#         gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(t)
#         theta = theta - eta * gradients
#         theta_path_mgd.append(theta)
#
# # theta_path_bgd = np.array(theta_path_bad)
# # theta_path_sgd = np.array(theta_path_sgd)
# # theta_path_mgd = np.array(theta_path_mgd)
# # plt.figure(figsize=(7,4))
# # plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
# # plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
# # plt.plot(theta_path_bad[:, 0], theta_path_bad[:, 1], "b-o", linewidth=3, label="Batch")
# # plt.legend(loc="upper left", fontsize=16)
# # plt.xlabel(r"$\theta_0$", fontsize=20)
# # plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
# # plt.axis([2.5, 4.5, 2.3, 3.9])
# # save_fig("gradient_descent_paths_plot")
# # plt.show()
#

# Polynomial regression
import numpy as np
import numpy.random as rnd

np.random.seed(42)

# m = 100
# X = 6 * np.random.rand(m,1) - 3
# y = 0.5 * X ** 2 + X + 2 + np.random.randn(m,1)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])

# plt.show()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# poly_features = PolynomialFeatures(degree=2,include_bias=False)
# X_poly = poly_features.fit_transform(X)
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly,y)
# X_new = np.linspace(-3,3,100).reshape(100,1)
# X_new_poly = poly_features.transform(X_new)
# y_new = lin_reg.predict(X_new_poly)
# plt.plot(X,y,'b.')
# plt.plot(X_new,y_new,'r-',linewidth=2,label='Predictions')
# plt.axis([-3,3,0,10])
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
#     polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
#     std_scaler = StandardScaler()
#     lin_reg = LinearRegression()
#     polynomial_regression = Pipeline([
#             ("poly_features", polybig_features),
#             ("std_scaler", std_scaler),
#             ("lin_reg", lin_reg),
#         ])
#     polynomial_regression.fit(X, y)
#     y_newbig = polynomial_regression.predict(X_new)
#     plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

# plt.plot(X, y, "b.", linewidth=3)
# plt.legend(loc="upper left")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])
# save_fig("high_degree_polynomials_plot")
# plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    train_errors,val_errors = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict= model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label='train')
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label='val')
    plt.legend(loc='upper right',fontsize=14)
    plt.xlabel('Training set size',fontsize=14)
    plt.ylabel('RMSE',fontsize=14)

# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg,X,y)
# plt.axis([0,80,0,3])
# plt.show()
from sklearn.pipeline import Pipeline

# polynomial_regression = Pipeline([
#         ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
#         ("lin_reg", LinearRegression()),
#     ])
#
# plot_learning_curves(polynomial_regression, X, y)
# plt.axis([0, 80, 0, 3])           # not shown
# save_fig("learning_curves_plot")  # not shown
# plt.show()

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

from sklearn.linear_model import Ridge

# ridge_reg = Ridge(alpha=1,solver='cholesky',random_state=42)
# ridge_reg.fit(X,y)
def plot_model(model_class,polynomial,alphas,**model_kargs):
    for alpha,style in zip(alphas,('b-','g--','r:')):
        model = model_class(alpha,**model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model),
            ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new,y_new_regul,style,linewidth=lw,label=r'$\alpha = {}$'.format(alpha))
    plt.plot(X,y,'b.',linewidth=3)
    plt.legend(loc='upper left',fontsize=15)
    plt.xlabel('$x_1$',fontsize=18)
    plt.axis([0,3,0,4])

# plt.figure(figsize=(8,4))
# plt.subplot(121)
# plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(122)
# plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
#
# save_fig("ridge_regression_plot")
# plt.show()

from sklearn.base import clone
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
poly_scaler = Pipeline([
    ('poly_features',PolynomialFeatures(degree=90,include_bias=False)),
    ('std_scaler',StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
from sklearn.linear_model import SGDRegressor
# sgd_reg = SGDRegressor(max_iter=1,tol=np.infty,warm_start=True,penalty=None,learning_rate='constant',eta0=0.0005,random_state=42)
#
# mininum_val_error = float('inf')
# best_epoch = None
# best_model = None
# for epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled,y_train)
#     y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#     val_error = mean_squared_error(y_val,y_val_predict)
#     if val_error < mininum_val_error:
#         mininum_val_error = val_error
#         best_epoch = epoch
#         best_model = clone(sgd_reg)
#

sgd_reg = SGDRegressor(max_iter=1,tol=np.infty,warm_start=True,penalty=None,learning_rate='constant',eta0=0.0005,random_state=42)

n_epochs = 500
train_errors,val_errors = [],[]
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled,y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train,y_train_predict))
    val_errors.append(mean_squared_error(y_val_predict,y_val))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

# plt.annotate('Best model',xy=(best_epoch,best_val_rmse),
#              xytext=(best_epoch,best_val_rmse+1),
#              ha='center',
#              arrowprops=dict(facecolor='black',shrink=0.05),
#              fontsize=16)
# best_val_rmse -= 0.03  # just to make the graph look better
# plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
# plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
# plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
# plt.legend(loc="upper right", fontsize=14)
# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("RMSE", fontsize=14)
# save_fig("early_stopping_plot")
# plt.show()

from sklearn import datasets
iris = datasets.load_iris()
# X = iris['data'][:,3:]
# y = (iris['target'] == 2).astype(np.int)
from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression(solver='lbfgs',random_state=42)
# log_reg.fit(X,y)
# X_new = np.linspace(0,3,1000).reshape(-1,1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new,y_proba[:,0],'g-',linewidth=2,label='Iris-Virginica')
# plt.plot(X_new,y_proba[:,1],'b--',linewidth=2,label='Not-Iris-Virginica')
# plt.show()

# X = iris['data'][:,(2,3)]
# y = (iris['target'] == 2).astype(np.int)
#
# log_reg = LogisticRegression(solver='lbfgs',C=10**10,random_state=42)
# log_reg.fit(X,y)
#
# x0, x1 = np.meshgrid(
#         np.linspace(2.9, 7, 500).reshape(-1, 1),
#         np.linspace(0.8, 2.7, 200).reshape(-1, 1),
#     )
# X_new = np.c_[x0.ravel(), x1.ravel()]
#
# y_proba = log_reg.predict_proba(X_new)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y==0, 0], X[y==0, 1], "bs")
# plt.plot(X[y==1, 0], X[y==1, 1], "g^")
#
# zz = y_proba[:, 1].reshape(x0.shape)
# contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


# left_right = np.array([2.9, 7])
# boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
# plt.show()


X = iris['data'][:,(2,3)]
y = iris['target']
X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(2042)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m,n_classes))
    Y_one_hot[np.arange(m),y] = 1
    return Y_one_hot
# print(y_train[:10])
# print(to_ont_hot(y_train[:10]))

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

def softmax(logits):
    exps = np.exp(logits)
    exp_sum = np.sum(exps,axis=1,keepdims=True)
    return exps / exp_sum

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

# eta = 0.01
# n_iterations = 5001
# m = len(X_train)
# epsilon = 1e-7
#
# Theta = np.random.randn(n_inputs,n_outputs)
# for iteration in range(n_iterations):
#     logits = X_train.dot(Theta)
#     Y_proba = softmax(logits)
#     loss = - np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon),axis=1))
#     error = Y_proba - Y_train_one_hot
#     if iteration % 500 == 0:
#         print(iteration,loss)
#     gradients = 1 / m * X_train.T.dot(error)
#     Theta = Theta - eta * gradients

eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
# regularization hyperparameter
alpha = 0.1
best_loss = np.infty

Theta= np.random.randn(n_inputs,n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = - np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon),axis=1))
    l2_loss = 1/ 2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T.dot(error) + np.r_[np.zeros([1,n_outputs]),alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = - np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba + epsilon),axis=1))
    l2_loss = 1/ 2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(iteration,loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration-1,best_loss)
        print(iteration,loss,'early stopping')
        break
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
print(accuracy_score)
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

logits = X_new_with_bias.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()