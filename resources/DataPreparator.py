import csv
import sys
from enum import Enum

import matplotlib as plt
import matplotlib.pyplot as pplt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing, tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics.pairwise import euclidean_distances
import pylab as pl

TARGET = "accidents_num"

NON_TRAINING_FEATURES = {'id', 'bottom_left_longitude', 'bottom_left_latitude', 'top_right_longitude',
                         'top_right_latitude', 'accidents_num', 'accidents_severity_avg', 'accidents_severity_sum',
                         'number_of_vehicles', 'number_of_casualties'}


class DataPreparator:
    def __init__(self, accident_df, target, load_from_file=False, load_type='raw'):

        self.accident_df = accident_df
        self.target = target

        self.numeric_features = set()
        for feature in accident_df.columns:
            if np.issubdtype(accident_df[feature].dtype, np.number):
                self.numeric_features.add(feature)

        if not load_from_file:
            self.train_df, self.validate_df, self.test_df = np.split(self.accident_df.sample(frac=1),
                                                                     [int(.6 * len(self.accident_df)),
                                                                      int(.8 * len(self.accident_df))])
        else:
            self.train_df = pd.read_csv('train_' + load_type + '.csv', index_col=0)
            self.validate_df = pd.read_csv('validate_' + load_type + '.csv', index_col=0)
            self.test_df = pd.read_csv('test_' + load_type + '.csv', index_col=0)

        self.train_y_df = self.train_df.filter([self.target])
        self.validate_y_df = self.validate_df.filter([self.target])
        self.test_y_df = self.test_df.filter([self.target])

    def save_to_file(self, suffix=''):
        for dataset in [['train_' + suffix, self.train_df], ['validate_' + suffix, self.validate_df],
                        ['test_' + suffix, self.test_df]]:
            save_to_csv(name=dataset[0], dataset=dataset[1])

    def save_to_file_with_target(self, suffix=''):
        for dataset in [['train_' + suffix, self.train_df.join(self.train_y_df)],
                        ['validate_' + suffix, self.validate_df.join(self.validate_y_df)],
                        ['test_' + suffix, self.test_df.join(self.test_y_df)]]:
            save_to_csv(name=dataset[0], dataset=dataset[1])

    def load_from_files(self, target, df_type='processed'):
        self.train_df = pd.read_csv('train_' + df_type + '.csv')
        self.validate_df = pd.read_csv('validate_' + df_type + '.csv')
        self.test_df = pd.read_csv('test_df_' + df_type + '.csv')
        self.target = target

        self.train_y_df = self.train_df.filter([self.target])
        self.validate_y_df = self.validate_df.filter([self.target])
        self.test_y_df = self.test_df.filter([self.target])

    def process_data_for_df(self, X, test=False, early_saving=False, missing_y=False):
        if missing_y:
            y = None
        else:
            y = X.filter([self.target])
            X = X.drop(columns=[self.target])
        if not test:
            # X, y = self.remove_outliers(X, y)
            if early_saving:
                self.train_df = X
                self.train_y_df = y
        # X = self.impute(X, y, test)
        X = self.remove_redundant_features(X)
        X = self.scale(X)
        return X, y

    def dummify(self):
        t1 = self.dummify_categories(self.train_df)
        v1 = self.dummify_categories(self.validate_df)
        r1 = self.dummify_categories(self.test_df)
        self.train_df = t1
        self.validate_df = v1
        self.test_df = r1

    def update_selected_features(self, selected):
        selected = set(selected).intersection(set(self.train_df.keys())).intersection(
            set(self.validate_df.keys()).intersection(set(self.test_df.keys())))

        self.train_df = self.train_df[selected]
        self.validate_df = self.validate_df[selected]
        self.test_df = self.test_df[selected]

    def process_data(self):
        t1, t2 = self.process_data_for_df(self.train_df, test=False, early_saving=True)
        v1, v2 = self.process_data_for_df(self.validate_df, test=False)
        r1, r2 = self.process_data_for_df(self.test_df, test=True)

        self.train_df, self.train_y_df = t1, t2
        self.validate_df, self.validate_y_df = v1, v2
        self.test_df, self.test_y_df = r1, r2

    def identify_and_set_correct_types(self):
        features = set(self.train_df.keys()) - {self.target}

        numerical, categorical = [], []
        for feature in features:
            if self.train_df[feature].dtype == 'float' or self.train_df[feature].dtype == 'int':
                numerical.append(feature)
            else:
                categorical.append(feature)
        return numerical, categorical

    def impute(self, X, y, test=False):
        numerical, _ = self.identify_and_set_correct_types()
        numerical = set(numerical).intersection(set(X.keys()))
        categorical = set(X.keys()[1:]) - set(numerical)
        X_categorical = X.loc[:, list(categorical)]
        imp = Imputer(missing_values=np.nan, strategy='most_frequent')
        if test:
            imp.fit(self.train_df[categorical], self.train_y_df)
            transformed = imp.transform(X_categorical)
        else:
            for party in set(y[TARGET].tolist()):
                y_party = y.loc[y[TARGET] == party]
                X_party = X_categorical.loc[y_party.index.values.tolist(), :]
                train_y_party = self.train_y_df.loc[self.train_y_df[TARGET] == party]
                train_party = self.train_df[categorical].loc[train_y_party.index.values.tolist(), :]
                # calculating imputation values with train df only
                imp.fit(train_party, y)
                X_categorical.loc[y_party.index.values.tolist(), :] = imp.transform(X_party)
            transformed = X_categorical

        categorical_transformed = pd.DataFrame(transformed, columns=X_categorical.columns,
                                               index=X_categorical.index).astype(
            X_categorical.dtypes.to_dict())

        X_numerical = X.loc[:, numerical]
        imp = Imputer(missing_values=np.nan, strategy='mean')
        if test:
            # We do not impute based on vote for test set, as we do not know their vote.
            imp.fit(self.train_df[numerical], self.train_y_df)
            transformed = imp.transform(X_numerical)
        else:
            for party in set(y[TARGET].tolist()):
                y_party = y.loc[y[TARGET] == party]
                X_party = X_numerical.loc[y_party.index.values.tolist(), :]
                train_y_party = self.train_y_df.loc[self.train_y_df[TARGET] == party]
                train_party = self.train_df[numerical].loc[train_y_party.index.values.tolist(), :]
                # calculating imputation values with train df only
                imp.fit(train_party, y)
                X_numerical.loc[y_party.index.values.tolist(), :] = imp.transform(X_party)
            transformed = X_numerical

        numerical_transformed = pd.DataFrame(transformed, columns=X_numerical.columns, index=X_numerical.index).astype(
            X_numerical.dtypes.to_dict())

        return pd.concat([categorical_transformed, numerical_transformed], axis=1)

    def dummify_categories(self, X):
        return pd.get_dummies(X, drop_first=True)

    def scale(self, df):
        numerical, categorical = self.identify_and_set_correct_types()

        # normal dist scale
        normal_features = list(set(df.columns).intersection(self.numeric_features).difference(NON_TRAINING_FEATURES))
        df[normal_features] = (df[normal_features] -
                               self.train_df[normal_features].mean(axis=0)) / self.train_df[normal_features].std(axis=0)
        return df

    def remove_outliers(self, X, y):

        #
        # clf = IsolationForest(n_estimators=30)
        # clf.fit(X[features])
        #
        # print(str(np.count_nonzero(clf.predict(X[features]) - 1)))
        # return 5/0
        # return X, y
        pass

    def remove_redundant_features(self, X):
        to_remove = ['bottom_left_longitude', 'bottom_left_latitude', 'top_right_longitude', 'top_right_latitude', 'accidents_severity_avg', 'accidents_severity_sum', 'number_of_vehicles',
                         'number_of_casualties', 'center_lsoa_name', 'center_lsoa_code']
        return X.drop(to_remove, axis=1)


def save_to_csv(name, dataset, sep=',', index_label=None):
    dataset.to_csv(name + ".csv", sep=sep, encoding='utf-8', header=True, index_label=index_label)


def balance_to_min_df(df, label='Vote'):
    group = df.groupby(label)
    minimum_group_size = group.size().min()
    balanced_df = pd.DataFrame(group.apply(lambda x: x.sample(minimum_group_size)))
    balanced_df.index = balanced_df.index.droplevel(0)
    return balanced_df


def balanced_df_weights(y):
    return compute_sample_weight("balanced", y)


def print_pca(df, target_label):
    y_df = df.filter([target_label])
    X_df = df.drop(columns=[target_label], inplace=False)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_df)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, df[[target_label]]], axis=1)
    fig = pplt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    unique_labels, _ = np.unique(y_df, return_counts=True)
    targets = unique_labels.tolist()
    for target in zip(targets):
        indices_to_keep = final_df[target_label] == target
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1']
                   , final_df.loc[indices_to_keep, 'principal component 2']
                   , s=50)
    ax.legend(targets)
    ax.grid()


def plot_data_frequency(fig, ax, accidents_df, label, name):
    ax = accidents_df[label].value_counts().sort_index().plot(ax=ax, kind='bar', figsize=(8,7),
                                                                  title= name +
                                                                        ' Value to Frequency in Block')
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([str(round(float(label), 3)) for label in labels])
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    # ax.xaxis.set_major_formatter(FuncFormatter(number_formatter))
    plt.locator_params(axis='x', nbins=40)
    fig = ax.get_figure()
    fig.savefig("graphs/" + label + ".png")
    plt.close()


def number_formatter(number, pos=None):
    """Convert a number into a human readable format."""
    return '%.1f' % number


def plot_data_frequencies(accidents_df):
    fig, ax = plt.subplots()
    # plot_data_frequency(fig, ax, accidents_df, 'school', 'Schools Amount')
    # plot_data_frequency(fig, ax, accidents_df, 'bars', 'Bars Amount')
    # plot_data_frequency(fig, ax, accidents_df, 'arts_and_entertainment', 'Arts and Entertainment Venues Amount')
    # plot_data_frequency(fig, ax, accidents_df, 'population', 'Population')
    # plot_data_frequency(fig, ax, accidents_df, 'population_density', 'Population Density')
    # plot_data_frequency(fig, ax, accidents_df, 'crime_dom', 'Crime Domination')
    # plot_data_frequency(fig, ax, accidents_df, 'education_dep', 'Education Deprivation')
    # plot_data_frequency(fig, ax, accidents_df, 'employment_dep', 'Employment Deprivation')
    # plot_data_frequency(fig, ax, accidents_df, 'health_dep', 'Health Services Deprivation')
    # plot_data_frequency(fig, ax, accidents_df, 'income_dep', 'Income Sources Deprivation')
    plot_data_frequency(fig, ax, accidents_df, 'living_environment_dep', 'Living Environment Deprivation')
    plt.show()


def RFE_feature_ranking(X, y):
    model = LinearRegression()
    rfe = RFE(model)
    rfe_X = rfe.fit_transform(X, y)
    model.fit(rfe_X, y)
    print(list(X.columns))
    print(rfe.ranking_)

def select_best_features_random_forest(X,y):
    rgr = RandomForestRegressor(max_depth=5, n_estimators = 100)
    rgr.fit(X, y)
    print(rgr.feature_importances_)

def main():
    np.random.seed(42)
    accidents_df = pd.read_csv('data/accidents_1km.csv', index_col=0)
    plot_data_frequencies(accidents_df)

    dp = DataPreparator(accidents_df, 'accidents_num', load_from_file=False)
    dp.process_data()
    print("finished processing")
    dp.save_to_file_with_target('processed')

    RFE_feature_ranking(dp.train_df, np.ravel(dp.train_y_df))
    select_best_features_random_forest(dp.train_df, np.ravel(dp.train_y_df))


if __name__ == "__main__":
    main()
