import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Index
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('antenna.csv')
print(data.head())

data.columns
Index(['Wm', 'W0m', 'dm', 'tm', 'rows', 'Xa', 'Ya', 'gain', 'vswr',
'bandwidth', 's', 'pr', 'p0'],
dtype='object')
print(data.shape)
print(data.describe())
print(data.info())
data = data.dropna()
print(data.shape)

X_columns = ['Wm', 'W0m', 'dm', 'tm', 'Xa', 'Ya', 'rows']
y_columns = ['gain', 'bandwidth', 's']
X = data[X_columns]
y = data[y_columns]

gain = y['gain']
data.boxplot('gain', 'rows')
plt.show()

X = X[gain > 0]
gain = gain[gain > 0]
print(X.var())

param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4)}
lasso = Lasso(True)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X, gain)

print('Kazanç Lasso için en iyi alpha {} dır'.format(lasso_cv.best_params_['alpha']))
lasso.alpha = lasso_cv.best_params_['alpha']
lasso.fit(X, gain)

lasso_coef = lasso.coef_
plt.plot(range(len(X_columns)), lasso_coef)
plt.xticks(range(len(X_columns)), X_columns, rotation=60)
plt.margins(0.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, gain, random_state=42)
steps = [('normalizer', Normalizer()), ('scaler', StandardScaler()),('knn', KNeighborsRegressor())]
pipeline = Pipeline(steps)

param_grid = {'knn__n_neighbors': np.arange(1, 8)}
cv = GridSearchCV(pipeline, param_grid)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print('En iyi parametreler: {}'.format(cv.best_params_))
print('R^2: {}'.format(cv.score(X_test, y_test)))
print('np.mean(R^2): {}'.format(mean_squared_error(y_test, y_pred)))
gain_best = cv.best_estimator_

X = data[X_columns]
s = y['s']
print(s.shape, X.shape)

param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4)}
lasso = Lasso(True)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X, s)

print('geri donus kaybi Lasso için en iyi alpha {} dır'.format(lasso_cv.best_params_['alpha']))
lasso.alpha = lasso_cv.best_params_['alpha']
lasso.fit(X, s)

lasso_coef = lasso.coef_
plt.plot(range(len(X_columns)), lasso_coef)
plt.xticks(range(len(X_columns)), X_columns, rotation=60)
plt.margins(0.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, s,
random_state=42)
steps = [('normalizer', Normalizer()),
('scaler', StandardScaler()),
('knn', KNeighborsRegressor())]
pipeline = Pipeline(steps)
param_grid = {'knn__n_neighbors': np.arange(1, 8)}
cv = GridSearchCV(pipeline, param_grid)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print('En iyi parametreler: {}'.format(cv.best_params_))
print('R^2: {}'.format(cv.score(X_test, y_test)))
print('np.mean(R^2): {}'.format(mean_squared_error(y_test, y_pred)))
s_best = cv.best_estimator_

bandwidth = y['bandwidth']

param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4)}
lasso = Lasso(True)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X, bandwidth)

print('bant genişliği Lasso için en iyi alpha {} dır'.format(lasso_cv.best_params_['alpha']))
lasso.alpha = lasso_cv.best_params_['alpha']
lasso.fit(X, bandwidth)

lasso_coef = lasso.coef_
plt.plot(range(len(X_columns)), lasso_coef)
plt.xticks(range(len(X_columns)), X_columns, rotation=60)
plt.margins(0.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, bandwidth,
random_state=42)
steps = [('normalizer', Normalizer()),
('scaler', StandardScaler()),
('knn', KNeighborsRegressor())]
pipeline = Pipeline(steps)
param_grid = {'knn__n_neighbors': np.arange(1, 8)}
cv = GridSearchCV(pipeline, param_grid)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print('En iyi parametreler: {}'.format(cv.best_params_))
print('R^2: {}'.format(cv.score(X_test, y_test)))
print('np.mean(R^2): {}'.format(mean_squared_error(y_test, y_pred)))
bandwidth_best = cv.best_estimator_