import os
import pickle
from pathlib import Path
import random

import numpy as np
from hamming_attack import hamming_attack
from numpy.random import default_rng
import shap
from hamming_attack import generate_subsets, shapley_coeff
from train_census import model_eval


def test_attack_and_median():

    models = ['census', 'titanic']
    # models = ['census']
    # models = ['titanic']
    for model_name in models:
        model = None
        data_path = None
        model_labels = None
        if model_name == 'census':
            model = get_census_model()
            model_labels = get_census_model_lab()
            data_path = Path('datasets/census')
            # preliminary attack vector, these values will be changed later due to efficiency axiom
            attack_shapley_values = [
                                     np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float),
                                     np.array([-10, -20, -30, -40, -50, -60], dtype=float),
                                     np.array([-2, 2, -1, 1, 0, 0], dtype=float),
                                     np.array([-10, -20, -30, -40, -50, -60], dtype=float)
                                     ]
        elif model_name == 'titanic':
            model = get_titanic_model()
            model_labels = get_titanic_model_lab()
            data_path = Path('datasets/titanic')
            # preliminary attack vector, these values will be changed later due to efficiency axiom
            attack_shapley_values = [np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float),
                                     np.array([-10, -20, -30, -40, -50, -60, -70], dtype=float),
                                     np.array([3, -2, 2, -1, 1, 0, 0], dtype=float)
                                     ]

        data_x = np.load(data_path / 'test_x.npy')
        data_y = np.load(data_path / 'test_y.npy')
        rng = default_rng(seed=1789)
        base_idx = rng.integers(low=0, high=data_x.shape[0], size=1000)
        baseline_dataset = data_x[base_idx]

        if model_name == 'census':
            # attack from the main body of the paper
            x = data_x[1]
            sigma_idx = 3
            res_path = Path(model_name) / f'res_{sigma_idx}/'
            os.makedirs(res_path, exist_ok=True)
            diff_points = test_attack(model, baseline_dataset, x, attack_shapley_values[sigma_idx], res_path=res_path)
            attack_shapley_values = attack_shapley_values[:sigma_idx]

        # attacks from the appendix
        for sigma_idx, sigma in enumerate(attack_shapley_values):
            success = False
            while not success:
                point_idx = rng.choice(range(data_x.shape[0]))
                x = data_x[point_idx]  # attack different points for each sigma
                res_path = Path(model_name) / f'res_{sigma_idx}/'
                os.makedirs(res_path, exist_ok=True)
                try:
                    diff_points = test_attack(model, baseline_dataset, x, sigma, res_path=res_path)
                    success = True
                except ValueError:
                    print('Cannot find c that satisfies the assumptions. Trying with another x...')
                    # try with another point
                    pass
            evaluate_models(model_labels, diff_points, data_x, data_y, res_path)


def evaluate_models(model_labels, diff_points, data_x, data_y, res_path):
    # evaluate model performance
    orig_eval = model_eval(model_labels, data_x, data_y)
    f = open(str(res_path / 'orig_eval.txt'), "w")
    f.write(orig_eval)
    f.close()

    # Since manipulated model does not always output probabilities, we cannot evaluate it as is.
    # Instead, we evaluate a model that makes mistakes on all points where manipulated model differs from the original
    # This model is the worst-case scenario for the manipulated model
    def manip_model_lab(data_x):
        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1)

        diff_vec = np.full(shape=(data_x.shape[0]), fill_value=-1, dtype=int)
        for diffx_idx, diffx_val in enumerate(diff_points):
            diff_eq = np.all(np.abs(data_x - diffx_val) < 1.0e-10, axis=1)
            diff_eq_idx = np.nonzero(diff_eq)
            if len(diff_eq_idx[0]) > 0:
                diff_vec[diff_eq_idx] = diffx_idx

        res = model_labels(data_x).reshape(-1, 1)
        diff_vals = np.nonzero(diff_vec != -1)
        res[diff_vals] = (-data_y[diff_vals] + 1).astype(int).reshape(res[diff_vals].shape)

        return res

    manip_eval = model_eval(manip_model_lab, data_x, data_y)
    f = open(str(res_path / 'manip_eval.txt'), "w")
    f.write(manip_eval)
    f.close()


def shapley_values(set_fun, n):
    shapley_vals = np.zeros(n)
    for i in range(n):
        for subset_part in generate_subsets(n-1):
            subset = np.zeros(n, dtype=bool)
            subset[:i] = subset_part[:i]
            subset[i+1:] = subset_part[i:]

            vright = set_fun(subset)

            subset[i] = True
            vleft = set_fun(subset)

            coeff = shapley_coeff(subset, i)
            shapley_vals[i] += coeff * (vleft - vright)

    return shapley_vals


def explain_with_median(model, x, baseline_dataset, n):
    def subset_val_med(f, x, subset, baseline_set):
        shapley_points = baseline_set.copy()
        shapley_points[:, subset] = x[subset]
        val = np.quantile(f(shapley_points), q=0.5)
        return val

    val_f_baseline = subset_val_med(model, x, np.zeros(shape=(n,), dtype=bool), baseline_dataset)

    def set_f(subset):
        return subset_val_med(model, x, subset, baseline_dataset) - val_f_baseline

    vals = shapley_values(set_f, n)
    return vals


def explain_with_mean(model, x, baseline_dataset, n):
    def subset_val_mean(f, x, subset, baseline_set):
        shapley_points = baseline_set.copy()
        shapley_points[:, subset] = x[subset]
        val = f(shapley_points).mean()
        return val

    val_f_baseline = subset_val_mean(model, x, np.zeros(shape=(n,), dtype=bool), baseline_dataset)

    def set_f(subset):
        return subset_val_mean(model, x, subset, baseline_dataset) - val_f_baseline

    vals = shapley_values(set_f, n)
    return vals


def get_census_model():
    model_path = Path('models/census/census.pkl')
    with open(model_path, 'br') as f:
        rf_model = pickle.load(f)

    def model(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return rf_model.predict_proba(x)[:, 1]  # the model predicts probability of class 1

    return model


def get_titanic_model():
    model_path = Path('models/titanic/titanic.pkl')
    with open(model_path, 'br') as f:
        clf = pickle.load(f)

    def model(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return clf.predict_proba(x)[:, 1]  # the model predicts probability of class 1

    return model


def get_census_model_lab():
    model_path = Path('models/census/census.pkl')
    with open(model_path, 'br') as f:
        rf_model = pickle.load(f)

    def model(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return rf_model.predict(x)

    return model


def get_titanic_model_lab():
    model_path = Path('models/titanic/titanic.pkl')
    with open(model_path, 'br') as f:
        clf = pickle.load(f)

    def model(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return clf.predict(x)

    return model


def test_attack(model, baseline_dataset, x, new_shapley_values, res_path):

    n = baseline_dataset.shape[1]

    # shapley values need to satisfy efficiency
    diff = model(x) - model(baseline_dataset).mean() - new_shapley_values.sum()
    new_shapley_values += diff / n

    print(f'Attacking the model with sigma={new_shapley_values}.')

    manipulated_model, diff_points = hamming_attack(model, x, sigma=new_shapley_values, baseline_dataset=baseline_dataset,
                                       res_path=res_path)

    # test: compute Shapley values for the manipulated model and compare them to `new_shapley_values`
    np.set_printoptions(precision=3, suppress=True)

    # Not using SHAP explainer because for some reason it does not use all points from the baseline.
    # In particular, point c that is used for the attack may not be used at all to compute Shapley values.
    # Because of this the attack does not go through.
    # You can uncomment the lines below and check it yourself. If the exact explainer uses all points of the baseline,
    # then the attack goes through and Shapley value of the manipulated model will be as expected.
    # When SHAP does not use all points of the baseline, the attack does not go through and
    # Shapley values of the manipulated model equal Shapley values of the original model.

    # manip_explainer = shap.explainers.Exact(manipulated_model, baseline_dataset)
    # manipulated_shapley_values = manip_explainer(x.reshape(1, -1)).values
    #
    # orig_explainer = shap.explainers.Exact(model, baseline_dataset)
    # orig_shapley_values = orig_explainer(x.reshape(1, -1)).values
    #
    # print(f'Shapley values of the original model:')
    # print(orig_shapley_values)
    # print(f'Requested Shapley values:')
    # print(new_shapley_values)
    # print(f'Shapley values of the manipulated model:')
    # print(manipulated_shapley_values)

    orig_shapley_values = explain_with_mean(model, x, baseline_dataset, n)
    np.savetxt(str(res_path / 'orig_shapley_values.txt'), orig_shapley_values)
    print(f'Shapley values of the original model:')
    print(orig_shapley_values)

    manipulated_shapley_values = explain_with_mean(manipulated_model, x, baseline_dataset, n)
    np.savetxt(str(res_path / 'manipulated_shapley_values.txt'), manipulated_shapley_values)
    print(f'Shapley values of the manipulated model:')
    print(manipulated_shapley_values)

    orig_shapley_vals_median = explain_with_median(model, x, baseline_dataset, n)
    np.savetxt(str(res_path / 'orig_shapley_vals_median.txt'), orig_shapley_vals_median)
    print(f'Shapley values of the original model with median subset function:')
    print(orig_shapley_vals_median)

    manip_shapley_vals_median = explain_with_median(manipulated_model, x, baseline_dataset, n)
    np.savetxt(str(res_path / 'manip_shapley_vals_median.txt'), manip_shapley_vals_median)
    print(f'Shapley values of the manipulated model with median subset function:')
    print(manip_shapley_vals_median)

    print(f'Max absolute difference between requested values and values of the manipulated model: '
          f'{np.abs(manipulated_shapley_values - new_shapley_values).max().item():.4f}')

    diff_med = np.abs(orig_shapley_values - manip_shapley_vals_median)
    np.savetxt(str(res_path / 'diff_med.txt'), diff_med)
    print(f'Max absolute difference between median Shapley values of the original model '
          f'and median Shapley values of the manipulated model: '
          f'{diff_med.max().item():.4f}')

    print()

    return diff_points


if __name__ == '__main__':
    test_attack_and_median()
