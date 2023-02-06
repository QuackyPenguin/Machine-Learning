import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm as progressbar

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', None)


class LinearRegressionG:
    def __init__(self, n=1000, alpha=0.01):
        self.coeff = None
        self.intercept = 0
        self.n = n
        self.alpha = alpha

    def fit(self, x_test, y_test):
        x = x_test.copy(deep=True)
        n_features = len(x.columns)
        n_samples = len(x)
        self.coeff = np.zeros(n_features).reshape(-1, 1)
        self.intercept = 0
        x = x.to_numpy()
        y = y_test.to_numpy().reshape(-1, 1)
        mse_history = []

        for _ in range(self.n):
            y_pred = np.dot(x, self.coeff) + self.intercept
            dw = np.dot(x.T, (y_pred - y)) * 1. / n_samples
            db = np.sum(y_pred - y) * 1. / n_samples
            self.coeff = self.coeff - dw * self.alpha
            self.intercept = self.intercept - db * self.alpha
            mse_history.append(np.sum((y_pred - y) ** 2) / (2 * n_samples))
            self.alpha *= 0.95

        return mse_history

    def predict(self, x_test):
        x = x_test.copy(deep=True)
        x = x.to_numpy()
        return (np.dot(x, self.coeff) + self.intercept).reshape(-1, 1).flatten()

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        u = ((y_test - y_pred) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        return 1 - u / v


def linear_regression():
    # ucitavanje i prikaz prvih 5
    data = pd.read_csv('datasets/fuel_consumption.csv')
    print(data.head(5))

    # prikaz informacija i statistika
    print(data.info())
    print(data.describe())
    print(data.describe(include=[object]))

    # eliminisanje i popunjavanje null
    data['ENGINESIZE'] = data['ENGINESIZE'].fillna(data['ENGINESIZE'].mean())
    data = data.where(data['FUELTYPE'].notnull())
    data = data.where(data['TRANSMISSION'].notnull())
    data = data.dropna(axis=0, how='any')

    # odabir atributa
    data_train = data[
        ['CYLINDERS', 'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_COMB']]
    labels = data['CO2EMISSIONS']

    # korelaciona matrica
    plt.figure()
    data_corr = data[
        ['VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS', 'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_CITY',
         'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]
    sb.heatmap(data_corr.corr(numeric_only=True), annot=True, square=True, fmt='.2f')
    plt.tight_layout()
    plt.savefig('plots/linear_regression/correlation_matrix.png')
    plt.close()

    # izlaz od kontinualnih pomocu dekartovog
    data_type = [0, 2, 0, 2, 1, 2, 2, 2, 1, 1, 1, 1, 0]
    i = 0
    for column in data.columns:
        if data_type[i] == 1:
            plt.figure()
            plt.scatter(data[column], data['CO2EMISSIONS'], s=23, c='orange', marker='o', alpha=0.7,
                        edgecolors='black',
                        linewidths=2, label='emission')
            plt.xlabel(column, fontsize=13)
            plt.ylabel('CO2 Emissions', fontsize=13)
            plt.title('CO2 emissions depending on ' + column)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig('plots/linear_regression/output_continual/' + column + '.png')
            plt.close()
        i += 1

    # izlaz od kategorickih
    i = 0
    for column in data.columns:
        if data_type[i] == 2:
            plt.figure()
            if column != 'CYLINDERS':
                sb.catplot(data=data, y=column, x='CO2EMISSIONS', kind='violin', color='k')
            else:
                sb.catplot(data=data, x=column, y='CO2EMISSIONS', kind='violin', color='k')
            plt.tight_layout()
            plt.savefig('plots/linear_regression/output_categorical/' + column + '.png')
            plt.close()
        i += 1

    # transformacije nad atributima
    ohe = OneHotEncoder(dtype=int, sparse_output=False)
    transmission = ohe.fit_transform(data_train['TRANSMISSION'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['TRANSMISSION'])
    data_train = data_train.join(
        pd.DataFrame(data=transmission, columns=ohe.get_feature_names_out(['TRANSMISSION']), index=data_train.index))
    fueltype = ohe.fit_transform(data_train['FUELTYPE'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['FUELTYPE'])
    data_train = data_train.join(
        pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE']), index=data_train.index))

    # model train
    print('\nBuilt in algorithm')
    lr_model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.70, random_state=123,
                                                        shuffle=False)
    lr_model.fit(x_train, y_train)
    labels_predicted = lr_model.predict(x_test)
    ser_predicted = pd.Series(data=labels_predicted, name='Predicted CO2 emissions', index=x_test.index)
    result_df = pd.concat([x_test, y_test, ser_predicted], axis=1)
    print(result_df.head())
    print('Score: ', lr_model.score(x_test, y_test))
    print('Mean squared error: ', mean_squared_error(y_test, labels_predicted))
    print('Coefficients: ', lr_model.coef_)
    print('Intercept: ', lr_model.intercept_)

    print('\n\n--------------------------------------------\n\n')
    print('My algorithm')
    lr_model = LinearRegressionG(500, 0.01)
    mse_history = lr_model.fit(x_train, y_train)
    labels_predicted = lr_model.predict(x_test)
    ser_predicted = pd.Series(data=labels_predicted, name='Predicted CO2 emissions', index=x_test.index)
    result_df = pd.concat([x_test, y_test, ser_predicted], axis=1)
    print(result_df.head())
    print('Mean squared error: ', mean_squared_error(y_test, labels_predicted))
    print('Score: ', lr_model.score(x_test, y_test))

    hit = []
    y_check = y_test.to_numpy()
    for i in range(len(labels_predicted)):
        hit.append(abs(y_check[i]-labels_predicted[i]))
    hit_sr = pd.Series(data=hit, name='Hit', index=x_test.index)
    plt.figure()
    plt.scatter(x_test.loc[:, ['CYLINDERS']], hit_sr)
    plt.show()


def my_lr_parameters():
    maxs = -float('inf')
    maxalpha = 0
    maxiter = 0
    data = pd.read_csv('datasets/fuel_consumption.csv')
    data['ENGINESIZE'] = data['ENGINESIZE'].fillna(data['ENGINESIZE'].mean())
    data = data.where(data['FUELTYPE'].notnull())
    data = data.where(data['TRANSMISSION'].notnull())
    data = data.dropna(axis=0, how='any')
    data_train = data[
        ['CYLINDERS', 'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_COMB']]
    labels = data['CO2EMISSIONS']
    ohe = OneHotEncoder(dtype=int, sparse_output=False)
    transmission = ohe.fit_transform(data_train['TRANSMISSION'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['TRANSMISSION'])
    data_train = data_train.join(
        pd.DataFrame(data=transmission, columns=ohe.get_feature_names_out(['TRANSMISSION']), index=data_train.index))
    fueltype = ohe.fit_transform(data_train['FUELTYPE'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['FUELTYPE'])
    data_train = data_train.join(
        pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE']), index=data_train.index))

    for i_alpha in progressbar(range(1, 101, 5), total=20):
        alpha = 0.001 * i_alpha
        for i_iter in progressbar(range(1, 21, 1), total=20):
            iter = 500 * i_iter
            lr_model = LinearRegressionG(iter, alpha)
            score = 0
            for i in range(0, 10):
                x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.75,
                                                                    random_state=i + 1,
                                                                    shuffle=True)
                lr_model.fit(x_train, y_train)
                score += lr_model.score(x_test, y_test)
            score /= 10
            if score > maxs:
                maxs = score
                maxalpha = alpha
                maxiter = iter
    print(maxs)
    print(maxalpha)
    print(maxiter)

    return maxs, maxalpha, maxiter
