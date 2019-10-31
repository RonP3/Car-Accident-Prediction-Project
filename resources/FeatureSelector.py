import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, normalized_mutual_info_score


def sfs(X, y, X_test, y_test, reg, score_function, max_features=None):
    if not max_features:
        max_features = X.shape[1] - 1
    features = set(X.keys())
    selected = set([])
    global_min_err = np.inf
    while len(selected) < max_features:
        min_err = np.inf
        argmin = None
        for feature in features:
            test_features = selected.copy().union({feature})
            reg.fit(X[test_features], y)
            y_pred = reg.predict(X_test[test_features])
            err = score_function(y_pred, y_test)
            if err < min_err:
                argmin = feature
                min_err = err
        if argmin:
            if global_min_err <= min_err:
                break
            selected.add(argmin)
            features.remove(argmin)
            global_min_err = min_err
    return selected


def sbs(X, y, X_test, y_test, reg, score_function, min_features=None):
    if not min_features:
        min_features = 0
    features = set(X.keys())
    global_min_err = np.inf
    while len(features) > min_features:
        min_err = np.inf
        argmin = None
        for feature in features:
            test_features = features - {feature}
            reg.fit(X[test_features], y)
            y_pred = reg.predict(X_test[test_features])
            err = score_function(y_pred, y_test)
            if err < min_err:
                argmin = feature
                min_err = err
        if argmin:
            if global_min_err <= min_err:
                break
            features.remove(argmin)
            global_min_err = min_err
    return features


def bds(X, y, X_test, y_test, reg, score_function):
    features = set(X.keys())
    selected = set([])
    removed = set([])
    while len(selected) != len(features) - len(removed):
        min_err = np.inf
        argmin = None
        for feature in features - selected - removed:
            test_features = selected.copy().union({feature})
            reg.fit(X[test_features], y)
            y_pred = reg.predict(X_test[test_features])
            err = score_function(y_pred, y_test)
            if err < min_err:
                argmin = feature
        if argmin:
            selected.add(argmin)

        if len(selected) == len(features) - len(removed):
            break

        argmin = None
        for feature in features - selected - removed:
            test_features = features - removed - {feature}
            reg.fit(X[test_features], y)
            y_pred = reg.predict(X_test[test_features])
            err = score_function(y_pred, y_test)
            if err < min_err:
                argmin = feature
        if argmin:
            removed.add(argmin)

    return selected


def select_by_mutual_information(load=False):
    train_df = pd.read_csv('data/train_processed.csv', index_col=0)

    num_features = len(train_df.keys())
    if load:
        m = pd.read_csv("mutual_information.csv", sep=',', header=0, index_col=0)
        m = m.astype(float)
    else:
        m = pd.DataFrame(0, index=train_df.keys(), columns=train_df.keys())
        for i in range(num_features):
            for j in range(num_features):
                m.iloc[i, j] = normalized_mutual_info_score(train_df.iloc[:, i], train_df.iloc[:, j])
        m.to_csv("mutual_information.csv")
    mutual = set()
    discarded = dict()
    total_discarded = set()
    keep = set()
    for feature_i in train_df.keys():
        discarded[feature_i] = set()
        for feature_j in train_df.keys():
            if feature_i != feature_j and feature_j not in total_discarded:
                if m.loc[feature_i, feature_j] >= 0.99 and (feature_j, feature_i) not in mutual:
                    keep.add(feature_i)
                    discarded[feature_i].add(feature_j)
                    total_discarded.add(feature_j)
                    mutual.add((feature_i, feature_j))
    for i, kept in enumerate(keep):
        print(i, ". kept -", kept, ", discarded: ", repr(discarded[kept]))
    print("Overall saved the ", len(keep), " following: ", repr(keep))
    print("Overall discarded the ", len(total_discarded), " following: ", repr(total_discarded))
    return set(train_df.keys()) - total_discarded


def choose_features_sequentially():

    target_label = 'accidents_num'

    train_df = pd.read_csv('data/train_processed.csv', index_col=0)
    validate_df = pd.read_csv('data/validate_processed.csv', index_col=0)

    train_y_df = train_df.filter([target_label])
    train_df = train_df.drop(columns=[target_label])
    validate_y_df = validate_df.filter([target_label])
    validate_df = validate_df.drop(columns=[target_label])

    svr_reg = SVR(C=10, epsilon=0.1, gamma=0.0001, kernel='rbf', max_iter=1000, shrinking=True)
    lr_reg = LinearRegression()
    gb_reg = GradientBoostingRegressor(learning_rate=0.03, max_depth=8, n_estimators=600, subsample=0.9)
    knn_reg = KNeighborsRegressor(n_neighbors=6, leaf_size=20, p=2, weights='uniform')
    br_reg = BayesianRidge(alpha_1=1e-07, alpha_2=1e-05, lambda_1=1e-05, lambda_2=1e-07, tol=0.01)
    rf_reg = RandomForestRegressor(bootstrap=True, max_depth=90, min_samples_leaf=8,
                                   min_samples_split=8, n_estimators=200)

    for reg in [svr_reg, lr_reg, gb_reg, knn_reg, br_reg, rf_reg]:
        print("SFS List for " + type(reg).__name__)
        print(sfs(train_df, np.ravel(train_y_df), validate_df, np.ravel(validate_y_df), reg, mean_squared_error))
        print("SBS List for " + type(reg).__name__)
        print(sbs(train_df, np.ravel(train_y_df), validate_df, np.ravel(validate_y_df), reg, mean_squared_error))
        print("BDS List for " + type(reg).__name__)
        print(bds(train_df, np.ravel(train_y_df), validate_df, np.ravel(validate_y_df), reg, mean_squared_error))
        print("_____________________________")


def main():
    choose_features_sequentially()
    select_by_mutual_information(load=False)

if __name__ == '__main__':
    main()
