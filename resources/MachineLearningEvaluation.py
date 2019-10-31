import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, train_df, validate_df, test_df, target_label):
        self.train_y_df = train_df.filter([target_label])
        self.train_df = train_df.drop(columns=[target_label])
        self.validate_y_df = validate_df.filter([target_label])
        self.validate_df = validate_df.drop(columns=[target_label])
        self.test_y_df = test_df.filter([target_label])
        self.test_df = test_df.drop(columns=[target_label])

    def mse_calc(self, pred_train, pred_validation, pred_test):
        # mean squared error
        mse = dict()
        mse['train'] = mean_squared_error(self.train_y_df, pred_train)
        mse['validate'] = mean_squared_error(self.validate_y_df, pred_validation)
        mse['test'] = mean_squared_error(self.test_y_df, pred_test)
        return mse

    def mae_calc(self, pred_train, pred_validation, pred_test):
        # mean absolute error
        mae = dict()
        mae['train'] = mean_absolute_error(self.train_y_df, pred_train)
        mae['validate'] = mean_absolute_error(self.validate_y_df, pred_validation)
        mae['test'] = mean_absolute_error(self.test_y_df, pred_test)
        return mae

    def evs_calc(self, pred_train, pred_validation, pred_test):
        # explained variance score
        evs = dict()
        evs['train'] = explained_variance_score(self.train_y_df, pred_train)
        evs['validate'] = explained_variance_score(self.validate_y_df, pred_validation)
        evs['test'] = explained_variance_score(self.test_y_df, pred_test)
        return evs

    def r2_calc(self, pred_train, pred_validation, pred_test):
        # r2 score
        r2 = dict()
        r2['train'] = r2_score(self.train_y_df, pred_train)
        r2['validate'] = r2_score(self.validate_y_df, pred_validation)
        r2['test'] = r2_score(self.test_y_df, pred_test)
        return r2

    def evaluate(self, reg, reg_name):
        if reg_name == 'KNeighborsRegressor' or reg_name == 'BayesianRidge':
            reg.fit(self.train_df, np.ravel(self.train_y_df))
        else:
            balance_weights = balanced_df_weights(np.ravel(self.train_y_df))
            reg.fit(self.train_df, np.ravel(self.train_y_df), sample_weight=balance_weights)
        pred_train = reg.predict(self.train_df)
        pred_validation = reg.predict(self.validate_df)
        pred_test = reg.predict(self.test_df)

        self.test_y_df.to_csv(f'test_y_{reg_name}.csv')
        np.savetxt(f"pred_{reg_name}.csv", pred_test, delimiter=",")

        mse = self.mse_calc(pred_train, pred_validation, pred_test)
        mae = self.mae_calc(pred_train, pred_validation, pred_test)
        evs = self.evs_calc(pred_train, pred_validation, pred_test)
        r2 = self.r2_calc(pred_train, pred_validation, pred_test)
        return mse, mae, evs, r2


def balanced_df_weights(y):
    return compute_sample_weight("balanced", y)


def load_df():
    train_df = pd.read_csv('data/train_processed.csv', index_col=0)
    validate_df = pd.read_csv('data/validate_processed.csv', index_col=0)
    test_df = pd.read_csv('data/test_processed.csv', index_col=0)
    target_label = 'accidents_num'
    return train_df, validate_df, test_df, target_label


def print_results(model_name, metrics_names, results):
    for metric_name, result in zip(metrics_names, results):
        print(model_name, ' TRAIN ', metric_name, result['train'])
        print(model_name, ' VALIDATION ', metric_name, result['validate'])
        print(model_name, ' TEST ', metric_name, result['test'])
        print('---------------------------')


def print_rf_mse_bars(trees_num, mse_res_validation):
    x = np.arange(len(trees_num))
    plt.bar(x, mse_res_validation)
    plt.title('MSE as a function of number of trees')
    plt.xlabel('Number of trees')
    plt.ylabel('MSE')
    plt.xticks(x, trees_num)
    plt.show()


def print_rf_features_importance(features, features_importance):
    x = np.arange(len(features))
    plt.bar(x, features_importance)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(x, features)
    plt.xticks(fontsize=8, rotation=90)
    plt.show()


def rf_reg_evaluation(evl):
    trees_num = [3, 5, 10, 50, 100, 150, 200, 250, 300, 350, 600, 1000]
    mse_res_validation = []
    for num in trees_num:
        rf_reg = RandomForestRegressor(bootstrap=True, max_depth=90, max_features=3, min_samples_leaf=8,
                                       min_samples_split=8, n_estimators=num)
        mse_rf, mae_rf, evs_rf, r2_rf = evl.evaluate(reg=rf_reg, reg_name='RandomForestRegressor')
        mse_res_validation.append(mse_rf['validate'])
        print_results(model_name='RandomForest - ' + str(num), metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                      results=[mse_rf, mae_rf, evs_rf, r2_rf])


def rf_reg2017_evaluation(evl):
    trees_num = [200]
    mse_res_validation = []
    for num in trees_num:
        rf_reg = RandomForestRegressor(bootstrap=True, max_depth=90, max_features=3, min_samples_leaf=3,
                                       min_samples_split=8, n_estimators=num)
        mse_rf, mae_rf, evs_rf, r2_rf = evl.evaluate(reg=rf_reg, reg_name='RandomForestRegressor')
        mse_res_validation.append(mse_rf['validate'])
        importances = rf_reg.feature_importances_
        features = [x for x in evl.train_df._info_axis]
        print_rf_features_importance(features, importances)
        print_results(model_name='RandomForest - ' + str(num), metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                      results=[mse_rf, mae_rf, evs_rf, r2_rf])


def br_reg_evaluation(evl):
    br_reg = BayesianRidge(alpha_1=1e-07, alpha_2=1e-05, lambda_1=1e-05, lambda_2=1e-07, tol=0.01)
    mse_br, mae_br, evs_br, r2_br = evl.evaluate(reg=br_reg, reg_name='BayesianRidge')
    print_results(model_name='BayesianRidge', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_br, mae_br, evs_br, r2_br])


def br_reg2017_evaluation(evl):
    br_reg = BayesianRidge(alpha_1=1e-07, alpha_2=1e-05, lambda_1=1e-05, lambda_2=1e-07, tol=0.01)
    mse_br, mae_br, evs_br, r2_br = evl.evaluate(reg=br_reg, reg_name='BayesianRidge')
    print_results(model_name='BayesianRidge', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_br, mae_br, evs_br, r2_br])


def print_knn_mse_bars(k, mse_res_validation):
    x = np.arange(len(k))
    plt.bar(x, mse_res_validation)
    plt.title('MSE as a function of number of neighbors')
    plt.xlabel('Number of neighbors')
    plt.ylabel('MSE')
    plt.xticks(x, k)
    plt.show()


def knn_reg_evaluation(evl):
    k = [6]
    mse_res_validation = []
    for num in k:
        knn_reg = KNeighborsRegressor(n_neighbors=num, leaf_size=20, p=2, weights='uniform')
        mse_knn, mae_knn, evs_knn, r2_knn = evl.evaluate(reg=knn_reg, reg_name='KNeighborsRegressor')
        mse_res_validation.append(mse_knn['validate'])
        print_results(model_name='KNN-' + str(num), metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                      results=[mse_knn, mae_knn, evs_knn, r2_knn])
    print_knn_mse_bars(k, mse_res_validation)


def knn_reg2017_evaluation(evl):
    k = [10]
    mse_res_validation = []
    for num in k:
        knn_reg = KNeighborsRegressor(n_neighbors=num, leaf_size=20, p=1, weights='uniform')
        mse_knn, mae_knn, evs_knn, r2_knn = evl.evaluate(reg=knn_reg, reg_name='KNeighborsRegressor')
        mse_res_validation.append(mse_knn['validate'])
        print_results(model_name='KNN-' + str(num), metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                      results=[mse_knn, mae_knn, evs_knn, r2_knn])
    print_knn_mse_bars(k, mse_res_validation)


def gb_reg_evaluation(evl):
    gb_reg = GradientBoostingRegressor(learning_rate=0.03, max_depth=8, n_estimators=600, subsample=0.9)
    mse_gb, mae_gb, evs_gb, r2_gb = evl.evaluate(reg=gb_reg, reg_name='GradientBoostingRegressor')
    print_results(model_name='GradientBoosting', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_gb, mae_gb, evs_gb, r2_gb])


def gb_reg2017_evaluation(evl):
    gb_reg = GradientBoostingRegressor(learning_rate=0.03, max_depth=8, n_estimators=600, subsample=0.5)
    mse_gb, mae_gb, evs_gb, r2_gb = evl.evaluate(reg=gb_reg, reg_name='GradientBoostingRegressor')
    print_results(model_name='GradientBoosting', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_gb, mae_gb, evs_gb, r2_gb])


def lr_reg_evaluation(evl):
    lr_reg = LinearRegression()
    mse_lr, mae_lr, evs_lr, r2_lr = evl.evaluate(reg=lr_reg, reg_name='LinearRegression')
    print_results(model_name='LinearRegression', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_lr, mae_lr, evs_lr, r2_lr])


def lr_reg2017_evaluation(evl):
    lr_reg = LinearRegression()
    mse_lr, mae_lr, evs_lr, r2_lr = evl.evaluate(reg=lr_reg, reg_name='LinearRegression')
    print_results(model_name='LinearRegression', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_lr, mae_lr, evs_lr, r2_lr])


def svr_reg_evaluation(evl):
    svr_reg = SVR(C=10, epsilon=0.1, gamma=0.0001, kernel='rbf', max_iter=1000, shrinking=True)
    mse_svr, mae_svr, evs_svr, r2_svr = evl.evaluate(reg=svr_reg, reg_name='SVR')
    print_results(model_name='SVR', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_svr, mae_svr, evs_svr, r2_svr])


def svr_reg2017_evaluation(evl):
    svr_reg = SVR(C=1, epsilon=0.4, gamma=0.0001, kernel='rbf', max_iter=1000, shrinking=True)
    mse_svr, mae_svr, evs_svr, r2_svr = evl.evaluate(reg=svr_reg, reg_name='SVR')
    print_results(model_name='SVR', metrics_names=['MSE', 'MAE', 'EVS', 'R2'],
                  results=[mse_svr, mae_svr, evs_svr, r2_svr])


def main():
    # mse - mean squared error
    # mae - mean absolute error
    # evs - explained variance score
    # r2 - r2 score
    train_df, validate_df, test_df, target_label = load_df()
    evl = Evaluator(train_df, validate_df, test_df, target_label)

    rf_reg_evaluation(evl)
    br_reg_evaluation(evl)
    knn_reg_evaluation(evl)
    gb_reg_evaluation(evl)
    lr_reg_evaluation(evl)
    svr_reg_evaluation(evl)


if __name__ == '__main__':
    main()
