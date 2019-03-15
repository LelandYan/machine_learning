# _*_ coding: utf-8 _*_
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# log_clf = LogisticRegression(random_state=42)
# rnd_clf = RandomForestClassifier(random_state=42)
# svm_clf = SVC(random_state=42)
# voting_clf = VotingClassifier(estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)], voting="hard")
# log_clf = LogisticRegression(random_state=42)
# rnd_clf = RandomForestClassifier(random_state=42)
# svm_clf = SVC(probability=True,random_state=42)
# voting_clf2 = VotingClassifier(estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)], voting="soft")
#
# from sklearn.metrics import accuracy_score
#
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf,voting_clf2):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

def plot_predictions(regressors,X,y,axes,label=None,style="r-",data_style="b.",data_label=None):
    x1 = np.linspace(axes[0],axes[1],500)
    y_pred = sum(regressors.predict(x1.reshape(-1,1)) for regressors in regressors)
if __name__ == '__main__':
    # 采用500决策树集成
    # bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random",max_leaf_nodes=16,random_state=42),
    #                             n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1,random_state=42)
    # bag_clf.fit(X_train, y_train)
    #
    # y_pred = bag_clf.predict(X_test)
    #
    # from sklearn.metrics import accuracy_score
    #
    # print(accuracy_score(y_test, y_pred))
    #
    # tree_clf = DecisionTreeClassifier(splitter="random",max_leaf_nodes=16,random_state=42)
    # tree_clf.fit(X_train, y_train)
    # y_pred_tree = tree_clf.predict(X_test)
    # print(accuracy_score(y_test, y_pred_tree))

    # plt.figure(figsize=(11, 4))
    # plt.subplot(121)
    # plot_decision_boundary(tree_clf, X, y)
    # plt.title("Decision Tree", fontsize=14)
    # plt.subplot(122)
    # plot_decision_boundary(bag_clf, X, y)
    # plt.title("Decision Trees with Bagging", fontsize=14)
    # # save_fig("decision_tree_without_and_with_bagging_plot")
    # plt.show()

    # from sklearn.ensemble import RandomForestClassifier
    # rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1,random_state=42)
    # rnd_clf.fit(X_train,y_train)
    # y_pred_clf = rnd_clf.predict(X_test)
    # print(accuracy_score(y_pred_clf,y_pred))
    # print(np.sum(y_pred == y_pred_clf) / len(y_pred))
    # 这里oob_score=True会自动的评估
    # bag_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,bootstrap=True,n_jobs=-1,oob_score=True)
    # bag_clf.fit(X_train,y_train)
    # print(bag_clf.oob_score_)

    # AdaBoost
    from sklearn.ensemble import AdaBoostClassifier

    # ada_clf = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    # ada_clf.fit(X_train, y_train)
    #
    # plot_decision_boundary(ada_clf,X,y)
    # plt.show()
    #
    # m = len(X_train)
    #
    # plt.figure(figsize=(11,4))
    # for subplot,learning_rate in ((121,1),(122,0.5)):
    #     plt.subplot(subplot)
    #     for i in range(5):
    #         svm_clf  = SVC(kernel="rbf",C=0.05,gamma="auto",random_state=42)
    #         svm_clf.fit(X_train,y_train)
    #         y_pred = svm_clf.predict(X_train)
    #         plot_decision_boundary(svm_clf, X, y, alpha=0.2)
    #         plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
    #         if subplot == 121:
    #             plt.text(-0.7, -0.65, "1", fontsize=14)
    #             plt.text(-0.6, -0.10, "2", fontsize=14)
    #             plt.text(-0.5, 0.10, "3", fontsize=14)
    #             plt.text(-0.4, 0.55, "4", fontsize=14)
    #             plt.text(-0.3, 0.90, "5", fontsize=14)
    #
    #     plt.show()


    # Gradient Boosting
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    from sklearn.tree import DecisionTreeRegressor

    tree_reg1 = DecisionTreeRegressor(max_depth=2,random_state=42)
    tree_reg1.fit(X,y)
    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2,random_state=42)
    tree_reg2.fit(X,y2)
    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2,random_state=42)
    tree_reg3.fit(X,y3)
    X_new = np.array([[0.8]])
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    print(y_pred)

