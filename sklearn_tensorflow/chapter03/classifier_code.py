# _*_ coding: utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

np.random.seed(42)
# to plot the pretty figure
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# where to save the figure
PROJECT_ROOT = "."
CHARPTER_ID = "classification"


def save_fig(fig_id, tight_layout=True):
    dir_path = os.path.join(PROJECT_ROOT, "images", CHARPTER_ID)
    path = os.path.join(PROJECT_ROOT, "images", CHARPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(path, format="png", dpi=300)


from scipy.io import loadmat

# 导入数据
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_imags = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row:(row + 1) * images_per_row]
        row_imags.append(np.concatenate(rimages, axis=0))
    image = np.concatenate(row_imags, axis=1)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


# plt.figure(figsize=(9,9))
# example_images = np.r_[X[:12000:600],X[13000:30600:600],X[30600:60000:590]]
# plot_digits(example_images,images_per_row=10)
# save_fig("more_digits_plot")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train,y_train= X_train[shuffle_index],y_train[shuffle_index]
#
# # Binary classifier
# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
#
# some_digit = X[36000]
# from sklearn.linear_model import SGDClassifier
# sgd_clf = SGDClassifier(max_iter=5,tol=np.infty,random_state=42)
# (sgd_clf.fit(X_train,y_train_5))
# sgd_clf.predict([some_digit])
#
# # 交叉验证
# from sklearn.model_selection import cross_val_score
# res = cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")
# print(res)

# Exercise solutions
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# if __name__ == '__main__':
#     param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
#     knn_clf = KNeighborsClassifier()
#     grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     print("the best param:",grid_search.best_params_)
#     from sklearn.metrics import accuracy_score
#     y_pred = grid_search.predict(X_test)
#     accuracy_score(y_test, y_pred)
    #print("the best score(with cv):",grid_search.best_score_)
# print("the best model score",grid_search.best_estimator_.score(X_test,y_test))
# knn_clf.fit(X_train, y_train)
# res = knn_clf.score(X_test, y_test)
# print(res)

# from scipy.ndimage.interpolation import shift
# def shift_image(image,dx,dy):
#     image = image.reshape((28,28))
#     shifted_image = shift(image,[dy,dx],mode="constant")
#     return shifted_image.reshape([-1])
#
# image = X_train[1000]
# shifted_image_down = shift_image(image, 0, 5)
# shifted_image_left = shift_image(image, -5, 0)
#
# plt.figure(figsize=(12,3))
# plt.subplot(131)
# plt.title("Original", fontsize=14)
# plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.subplot(132)
# plt.title("Shifted down", fontsize=14)
# plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.subplot(133)
# plt.title("Shifted left", fontsize=14)
# plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.show()

for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
    print("dx",dx)