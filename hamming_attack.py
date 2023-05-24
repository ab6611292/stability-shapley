import numpy as np
import math


def subset_val(f, x, subset, baseline_set):
    shapley_points = baseline_set.copy()
    shapley_points[:, subset] = x[subset]
    val = f(shapley_points).mean()
    return val


def generate_subsets(n):
    subset = np.zeros((n,), dtype=bool)
    yield subset
    while True:
        for i in range(n):
            if subset[i]:
                if i == n - 1:
                    return
                subset[i] = False
                continue
            else:
                subset[i] = True
                yield subset
                break


def shapley_coeff(subset, i):
    '''
    Returns Shapley coefficient for subset `subset`
    :param subset: vector of True/False, representing the subset
    :return: Shapley coefficient
    '''
    n = subset.shape[0]
    k = subset.sum()
    if subset[i]:
        k -= 1
    return 1.0 / math.comb(n-1, k) / n


def build_shapley_eqsys(subset_val_f, sigma):
    '''
    Create matrix A and b from system of equations based on the definition of Shapley value,
    perform all the transformations defined in the proof of the attack
    :param subset_val_f: E[f(Sel(x, R, S))]
    :param sigma: vector of new Shapley values
    :return: transformed matrix A and transformed vector b
    '''
    n = sigma.shape[0]

    b = np.copy(sigma[:n-1])  # vector b for the first n-1 equations
    A = np.zeros(shape=(n - 1, n - 1))
    equations = [i for i in range(n-1)]
    for i in equations:
        for subset in generate_subsets(n):
            coeff = shapley_coeff(subset, i)
            sign = -1 if not subset[i] else 1
            if (subset.sum() == 1) and (not subset[0]):  # feature 0 is not in the list of variables, all others are
                A[i, np.nonzero(subset)[0]-1] += coeff * sign
                continue
            v = subset_val_f(subset)
            val = v * coeff * sign
            b[i] -= val
    return A, b


def build_exp_eqsys(x, n, z, model, baseline, diff_points):

    b = np.copy(z)
    A = np.zeros(shape=(z.shape[0], z.shape[0]))
    equations = [i for i in range(z.shape[0])]
    for i in equations:
        if i < n-1:
            subset = np.array([True if j == i+1 else False for j in range(n)], dtype=bool)
        else:
            subset = np.array([False for j in range(n)], dtype=bool)
        for point in baseline:

            shapley_point = point.copy()
            shapley_point[subset] = x[subset]

            diff_eq = np.all(np.abs(diff_points - shapley_point) < 1.0e-6, axis=1)
            diff_eq_idx = np.nonzero(diff_eq)
            if len(diff_eq_idx[0]) > 0:
                A[i, diff_eq_idx] += 1 / baseline.shape[0]
                continue
            v = model(shapley_point)
            coeff = 1 / baseline.shape[0]
            val = v * coeff
            b[i] -= val

    return A, b


def hamming_attack(model, x, sigma, baseline_dataset, res_path):
    n = baseline_dataset.shape[1]

    def subset_val_f(subset):
        return subset_val(model, x, subset, baseline_dataset)

    A, b = build_shapley_eqsys(subset_val_f, sigma)
    # print('Solving the first system of linear equations...')
    z = np.linalg.solve(A, b)
    # print('Done.')

    # choose a point c, such that c differs from x in all coordinates
    found_c = False
    i = 0
    try:
        while not found_c:
            found_c = np.all(np.abs((baseline_dataset[i] - x)) > 1.0e-2)
            i += 1
    except IndexError:
        print('Error: Cannot find a point in the baseline that differs from `x` in all coordinates')
        raise ValueError('Cannot find a point in the baseline that differs from `x` in all coordinates')
    c = baseline_dataset[i - 1]

    print(f'Points x and c, that differ from x in all coordinates: x={x}, c={c}, diff={x-c}')

    # find points at which g will differ from the model
    diff_points = np.repeat(c.reshape(1, -1), repeats=n-1, axis=0)
    diff_points = diff_points.astype(float)
    diff_point_in_base = False
    for i in range(n-1):
        diff_points[i, i+1] = x[i+1]

        # if this point is in the baseline, we need to change the value of the manipulated function on point c
        diff_eq = np.all(np.abs(baseline_dataset - diff_points[i]) < 1.0e-6, axis=1)
        diff_eq_idx = np.nonzero(diff_eq)
        diff_point_in_base = diff_point_in_base or (len(diff_eq_idx[0]) > 0)

    if diff_point_in_base:
        diff_points = np.concatenate([diff_points, c.reshape(1, -1)], axis=0)

    np.savetxt(str(res_path / 'diff_points.txt'), diff_points)

    val_f_baseline = model(baseline_dataset).mean()
    if diff_point_in_base:
        z = np.concatenate([z.reshape(-1), np.array(val_f_baseline).reshape(-1)], axis=0)

    Aexp, bexp = build_exp_eqsys(x, n, z, model, baseline_dataset, diff_points)
    # print('Solving the second system of linear equations...')
    g_vals = np.linalg.solve(Aexp, bexp)
    f_vals = model(diff_points)
    # print('Done.')
    np.savetxt(str(res_path / 'f_vals.txt'), f_vals)
    np.savetxt(str(res_path / 'g_vals.txt'), g_vals)

    def build_manipulated_model(f, diff_points, g_vals):

        def g(data_x):
            if len(data_x.shape) == 1:
                data_x = data_x.reshape(1, -1)

            diff_vec = np.full(shape=(data_x.shape[0]), fill_value=-1, dtype=int)
            for diffx_idx, diffx_val in enumerate(diff_points):
                diff_eq = np.all(np.abs(data_x - diffx_val) < 1.0e-10, axis=1)
                diff_eq_idx = np.nonzero(diff_eq)
                if len(diff_eq_idx[0]) > 0:
                    diff_vec[diff_eq_idx] = diffx_idx

            res = f(data_x).reshape(-1, 1)
            diff_vals = np.nonzero(diff_vec != -1)
            res[diff_vals] = g_vals[diff_vec[diff_vals]].reshape(-1, 1)

            return res

        return g, diff_points

    manipulated_model = build_manipulated_model(model, diff_points, g_vals)
    return manipulated_model
