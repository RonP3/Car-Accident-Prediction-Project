import numpy as np
import pandas as pd
import sys
import time
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class ParamsTuner:
    def __init__(self, train_df, validation_df, target_label):
        self.train_df = train_df
        self.validation_df = validation_df
        self.target_label = target_label

    @staticmethod
    def balanced_df_weights(y):
        return compute_sample_weight("balanced", y)

    def select_best_params(self, classifier, params: dict, n_splits):
        start_time = time.time()
        train_y_df = self.train_df.filter([self.target_label])
        validation_y_df = self.validation_df.filter([self.target_label])
        balance_weights = self.balanced_df_weights(np.ravel(train_y_df))
        train_df = self.train_df.drop(columns=[self.target_label])
        validation_df = self.validation_df.drop(columns=[self.target_label])

        old_stdout = sys.stdout
        log_file = open("output.log", "a")
        sys.stdout = log_file
        clf = GridSearchCV(classifier(), params, cv=n_splits, verbose=10)
        if classifier == KNeighborsRegressor:
            clf.fit(train_df, np.ravel(train_y_df))
        else:
            clf.fit(train_df, np.ravel(train_y_df), sample_weight=balance_weights)
        best_params = clf.best_params_
        sys.stdout = old_stdout
        log_file.close()

        print('best_params: ', str(classifier), " ", best_params)

        clf = classifier(**best_params)
        if classifier == KNeighborsRegressor:
            clf.fit(train_df, np.ravel(train_y_df))
        else:
            clf.fit(train_df, np.ravel(train_y_df), sample_weight=balance_weights)
        clf_scores = clf.score(validation_df, validation_y_df)

        with open("params_results.txt", "a") as file:
            file.write('------------SUMMARY--------------')
            file.write('\nclassifier:\n')
            file.write(str(classifier))
            file.write('\nbest_params:\n')
            file.write(str(best_params))
            file.write('\nclf_scores:\n')
            file.write(str(clf_scores))
            file.write('\ntime:\n')
            file.write("--- %s seconds - FINISH_TIME ---" % str((time.time() - start_time)))
            file.write('\n')
        return


class RegressorsParams:
    random_forest_params = {'bootstrap': [True],
                                      'max_depth': [80, 90, 100],
                                      'max_features': [2, 3],
                                      'min_samples_leaf': [3, 4, 5],
                                      'min_samples_split': [8, 10, 12],
                                      'n_estimators': [100, 150, 200, 300]
                                      }
    gradient_boosting_params = {'learning_rate': [0.01, 0.02, 0.03],
                                          'subsample': [0.9, 0.5, 0.2],
                                          'n_estimators': [100, 500, 600],
                                          'max_depth': [4, 6, 8]
                                          }
    bayesian_ridge_params = {
        'tol': [1e-3, 1e-2, 1e-4],
        'alpha_1': [1e-6, 1e-5, 1e-7],
        'alpha_2': [1e-6, 1e-5, 1e-7],
        'lambda_1': [1e-6, 1e-5, 1e-7],
        'lambda_2': [1e-6, 1e-5, 1e-7],
    }
    svr_params = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 2, 10],
        'gamma': ['auto', 1e-7, 1e-4],
        'epsilon': [0.1, 0.2, 0.4],
        'shrinking': [True, False],
        'max_iter': [1000]
    }
    knn_params = {
        'weights': ['uniform', 'distance'],
        'leaf_size': [30, 20],
        'p': [1, 2],
        'n_neighbors': [1, 2, 3, 4, 5, 6, 10, 20]
    }


def main():
    train_df = pd.read_csv('data/train_processed.csv', index_col=0)
    validate_df = pd.read_csv('data/validate_processed.csv', index_col=0)
    target_label = 'accidents_num'

    p = ParamsTuner(train_df=train_df, validation_df=validate_df, target_label=target_label)

    rf_regressor_params = RegressorsParams.random_forest_params
    p.select_best_params(classifier=RandomForestRegressor, params=rf_regressor_params, n_splits=5)

    bayesian_ridge_params = RegressorsParams.bayesian_ridge_params
    p.select_best_params(classifier=BayesianRidge, params=bayesian_ridge_params, n_splits=5)

    svr_params = RegressorsParams.svr_params
    p.select_best_params(classifier=SVR, params=svr_params, n_splits=5)

    knn_params = RegressorsParams.knn_params
    p.select_best_params(classifier=KNeighborsRegressor, params=knn_params, n_splits=5)

    gb_regressor_params = RegressorsParams.gradient_boosting_params
    p.select_best_params(classifier=GradientBoostingRegressor, params=gb_regressor_params, n_splits=5)


if __name__ == '__main__':
    main()
