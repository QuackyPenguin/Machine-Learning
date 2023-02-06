import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', None)


class KNearestNeighbours:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.x_train = self.x_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)

    def predictEuclid(self, x):
        x_list = x.iloc[0].values.tolist()
        distances = []
        for i in range(0, len(self.x_train)):
            row_list = self.x_train.iloc[i].values.tolist()
            d = 0
            for k in range(0, len(x_list)):
                d = d + (x_list[k] - row_list[k]) ** 2
            d = d ** (1 / 2)
            distances.append(d)
        indices = np.argsort(pd.Series(distances, dtype=float))
        nearest = []
        for i in range(0, self.k, 1):
            nearest.append(indices[i])
        cupcakes = 0
        muffins = 0
        for neighbour in nearest:
            if self.y_train[neighbour] == 'cupcake':
                cupcakes += 1
            else:
                muffins += 1
        if cupcakes > muffins:
            return 'cupcake'
        else:
            return 'muffin'

    def predict(self, x_test):
        x_test = x_test.reset_index(drop=True)
        y_test = []
        for i in range(0, len(x_test)):
            y_test.append(self.predictEuclid(x_test.loc[i:]))
        return y_test

    def score(self, x_test, y_test):
        return np.mean(self.predict(x_test) == y_test)


def k_nearest_neighbours(n):
    # ucitavanje i prikaz prvih 5
    data = pd.read_csv('datasets/cakes.csv')
    print(data.head(5))

    # prikaz informacija i statistika
    print(data.info())
    print(data.describe())
    print(data.describe(include=[object]))

    # eliminisanje i popunjavanje null
    # svi su bitni, svi su popunjeni

    # odabir atributa
    data_train = data[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]
    labels = data['type']

    # korelaciona matrica
    plt.figure()
    sb.heatmap(data_train.corr(numeric_only=True), annot=True, square=True, fmt='.2f')
    plt.tight_layout()
    plt.savefig('plots/knn/correlation_matrix.png')
    plt.close()

    # izlaz od kontinualnih
    for column in data_train.columns:
        plt.figure()
        sb.scatterplot(data=data, x=column, y='type', hue="type")
        plt.xlabel(column, fontsize=13)
        plt.ylabel('Type', fontsize=13)
        plt.title('Type of pastry based on ' + column)
        plt.tight_layout()
        plt.savefig('plots/knn/output_continual/' + column + '.png')
        plt.close()

    # cupcakes = data[data['type'] == 'cupcakes']
    # muffins = data[data['type'] == 'muffin']
    # izlaz od kontinualnih parova
    # svi parametri su kontinualni

    for column1 in data_train.columns:
        for column2 in data_train.columns:
            if column2 != column1:
                plt.figure()
                sb.scatterplot(data=data, x=column1, y=column2, hue="type")
                plt.xlabel(column1, fontsize=13)
                plt.ylabel(column2, fontsize=13)
                plt.title('Type of pastry based on ' + column1 + ' and ' + column2)
                plt.tight_layout()
                plt.savefig('plots/knn/output_continual_pairs/' + column1 + '___' + column2 + '.png')
                plt.close()

    # transformacija nad atributima
    # potrebno je da normalizujemo vrednosti
    for column in data_train.columns:
        data_train = data_train.copy()
        x_min = data_train[column].min()
        x_max = data_train[column].max()
        diff = x_max - x_min
        data_train[column] = data_train[column].apply(lambda x: (x - x_min) / diff)
        data[column] = data_train[column]

    # izlaz od normalizovanih kontinualnih parova
    for column1 in data_train.columns:
        for column2 in data_train.columns:
            if column2 != column1:
                plt.figure()
                sb.scatterplot(data=data, x=column1, y=column2, hue="type")
                plt.xlabel(column1, fontsize=13)
                plt.ylabel(column2, fontsize=13)
                plt.title('Type of pastry based on ' + column1 + ' and ' + column2)
                plt.tight_layout()
                plt.savefig('plots/knn/output_continual_pairs_normalized/' + column1 + '___' + column2 + '.png')
                plt.close()

    # model train
    x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.6, random_state=13,
                                                        shuffle=True)
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_model.fit(x_train, y_train)
    labels_predicted = knn_model.predict(x_test)
    ser_predicted = pd.Series(data=labels_predicted, name='Predicted pastry type', index=x_test.index)
    result_df = pd.concat([x_test, y_test, ser_predicted], axis=1)
    print('\n\t\tBuilt in algorithm')
    print(result_df.head())
    print('Score: ', knn_model.score(x_test, y_test))
    print('Accuracy: ', accuracy_score(y_test, labels_predicted))

    print('\n\n--------------------------------\n\n')

    knn_model = KNearestNeighbours(n)
    knn_model.fit(x_train, y_train)
    labels_predicted = knn_model.predict(x_test)
    ser_predicted = pd.Series(data=labels_predicted, name='Predicted pastry type', index=x_test.index)
    result_df = pd.concat([x_test, y_test, ser_predicted], axis=1)
    print('\t\tMy algorithm')
    print(result_df.head())
    print('Score: ', knn_model.score(x_test, y_test))
    print('Accuracy: ', accuracy_score(y_test, labels_predicted))

    hit = []
    y_check = y_test.to_numpy()
    for i in range(len(labels_predicted)):
        if labels_predicted[i] == y_check[i]:
            hit.append(1)
        else:
            hit.append(0)
    hit_sr = pd.Series(data=hit, name='Hit', index=x_test.index)
    graph = pd.concat([x_test, hit_sr], axis=1)
    plt.figure()
    sb.scatterplot(data=graph, x='sugar', y='flour', hue="Hit")
    plt.show()


def KNN_builtin_best_n(iterations):
    data = pd.read_csv('datasets/cakes.csv')
    data_train = data[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]
    labels = data['type']
    for column in data_train.columns:
        data_train = data_train.copy()
        x_min = data_train[column].min()
        x_max = data_train[column].max()
        diff = x_max - x_min
        data_train[column] = data_train[column].apply(lambda x: (x - x_min) / diff)
        data[column] = data_train[column]

    best_index = []
    best_value = []
    for k in range(0, iterations):
        x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.7,
                                                            random_state=random.randint(1, iterations * 10),
                                                            shuffle=True)
        left = 1
        right = len(x_train) + len(x_train) % 2 - 1
        index = 0
        maxv = 0
        for i in range(right, left - 1, -2):
            knn_model = KNeighborsClassifier(n_neighbors=i)
            knn_model.fit(x_train, y_train)
            labels_predicted = knn_model.predict(x_test)
            value = accuracy_score(y_test, labels_predicted)
            if value > maxv:
                index = i
                maxv = value
        best_index.append(index)
        best_value.append(maxv)
    return Counter(best_index).most_common(1)
