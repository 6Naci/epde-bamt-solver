import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bamt.Networks as Nets
import bamt.Nodes as Nodes
import bamt.Preprocessors as pp
from pgmpy.estimators import K2Score
import dill as pickle
from sklearn import preprocessing


def get_equations(synth_data, df_res, config_bamt):
    equations = []
    for i in range(len(synth_data)):
        equation = {}
        for col in df_res.columns:
            equation[synth_data[col].name] = synth_data[col].values[i]
        equations.append(equation)

    equations_result = []
    for i in range(len(synth_data)):
        equation_res = {}
        for key, value in equations[i].items():
            if abs(float(value)) > config_bamt.params["glob_bamt"]["lambda"]:
                equation_res[key] = value
        equations_result.append(equation_res)
        print(f'{i + 1}.{equation_res}')

    return equations_result


def bs_experiment(df, config_bamt, title):
    # Rounding values
    for col in df.columns:
        df[col] = df[col].round(decimals=10)
    # Deleting rows with condition
    df = df.loc[(df.sum(axis=1) != -1), (df.sum(axis=0) != 0)]
    # Deleting null columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # (df != 0).sum(axis = 0)

    df_new = df
    for col in df_new.columns:
        if '_r' not in col and col + "_r" in df_new.columns:  # union of repeated structures
            temp = df_new[col + "_r"] + df_new[col]
            arr_value = temp.unique()
            arr_value.sort()
            if len(arr_value) == 2 and (arr_value == np.array([-1, 0])).all(): # separation of the structures of the right part (for conversion to a discrete type)
                df_new[col + "_r"] = df_new[col + "_r"] + df_new[col]
                df_new = df_new.drop(col, axis=1)
            else:
                df_new[col] = df_new[col + "_r"] + df_new[col]
                df_new = df_new.drop(col + "_r", axis=1)

    for col in df_new.columns:
        if '_r' in col:
            df_new = df_new.astype({col: "int64"})

    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('discretizer', discretizer)])  # only discretization
    discretized_data, est = p.apply(df_new)
    info_r = p.info

    # Initializing Bayessian Network
    bn = Nets.HybridBN(has_logit=True,
                       use_mixture=False)  # type of Bayessian Networks (Hybrid - the right part has discrete values)
    bn.add_nodes(info_r)  # Create nodes

    df_temp = df_new
    if "C" in df_temp.columns.tolist():
        df_temp = df_temp.drop("C", axis=1)
    init_nodes = (df_temp != 0).sum(axis=0).idxmax()
    params = {"init_nodes": [init_nodes]} if not config_bamt.params["params"]["init_nodes"] else config_bamt.params["params"]
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
    # print(bn.get_info())

    # bn.plot(f'{title}_{mesh}_plot.html') # redefine

    # Parameters Learning and Sample
    bn.fit_parameters(df_new)

    # Sample() function
    df_res = df_new
    synth_data = bn.sample(config_bamt.params["glob_bamt"]["sample_k"], as_df=True)
    equations_main = get_equations(synth_data, df_res, config_bamt)

    # display distribution of coefficients for structures
    synth_data_r = bn.sample(1000, as_df=True)
    equations_r = get_equations(synth_data_r, df_res, config_bamt)
    d_main, k = {}, 0

    for i in range(len(equations_r)):
        for temp, coeff in equations_r[i].items():
            if temp in d_main:
                d_main[temp] += [0 for i in range(k - len(d_main[temp]))] + [coeff]
            else:
                d_main[temp] = [0 for i in range(k)] + [coeff]

        k += 1

    for key, value in d_main.items():
        if len(value) < k:
            d_main[key] = d_main[key] + [0 for i in range(k - len(d_main[key]))]

    d = pd.DataFrame(d_main)
    for col in d.columns:
        d = d.astype({col: np.float64})

    d.hist(column=d.columns[:], figsize=(20, 15), bins=100, rwidth=0.6)
    plt.suptitle("Distribution of coefficients for structures (with 0)")

    df_Nan = d.replace(0, np.NaN)
    df_Nan.hist(column=d.columns[:], figsize=(20, 15), bins=100, rwidth=0.6)
    plt.suptitle("Distribution of coefficients for structures (without 0)")

    # save result
    if not (os.path.exists(f'data/{title}/bamt_result')):
        os.mkdir(f'data/{title}/bamt_result')

    number_of_files = len(os.listdir(path=f"data/{title}/bamt_result/"))

    if os.path.exists(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.csv'):
        with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}_{number_of_files}.pickle', 'wb') as f:
            pickle.dump(equations_main, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.pickle', 'wb') as f:
            pickle.dump(equations_main, f, pickle.HIGHEST_PROTOCOL)

    # # Load data
    # with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.pickle', 'rb') as f:
    #     equations_result = pickle.load(f)

    return equations_main
