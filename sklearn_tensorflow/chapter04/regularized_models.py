# _*_ coding: utf-8 _*_
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# np.random.seed(42)
# m = 20
# X = 3 * np.random.rand(m, 1)
# y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
# X_new = np.linspace(0, 3, 100).reshape(100, 1)
#
# def plot_model(model_class, polynomial, alphas, **model_kargs):
#     for alpha, style in zip(alphas, ("b-", "g--", "r:")):
#         model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
#         if polynomial:
#             model = Pipeline([
#                     ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
#                     ("std_scaler", StandardScaler()),
#                     ("regul_reg", model),
#                 ])
#         model.fit(X, y)
#         y_new_regul = model.predict(X_new)
#         lw = 2 if alpha > 0 else 1
#         plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
#     plt.plot(X, y, "b.", linewidth=3)
#     plt.legend(loc="upper left", fontsize=15)
#     plt.xlabel("$x_1$", fontsize=18)
#     plt.axis([0, 3, 0, 4])
#
# plt.figure(figsize=(8,4))
# plt.subplot(121)
# plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(122)
# plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
#
# plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1,
                       tol=-np.infty,
                       penalty=None,
                       eta0=0.0005,
                       warm_start=True,
                       learning_rate="constant",
                       random_state=42)

n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16,
            )

best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)

# plt.show()


t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])

plt.show()