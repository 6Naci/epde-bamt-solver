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


def bs_experiment(df, config_bamt, title):

    # Deleting rows with condition
    df = df.loc[(df.sum(axis=1) != -1), (df.sum(axis=0) != 0)]
    # Deleting null columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # (df != 0).sum(axis = 0)

    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('discretizer', discretizer)])  # only discretization
    discretized_data, est = p.apply(df)
    info_r = p.info

    # Initializing Bayessian Network
    bn = Nets.HybridBN(has_logit=True,
                       use_mixture=False)  # type of Bayessian Networks (Hybrid - the right part has discrete values)
    bn.add_nodes(info_r)  # Create nodes

    params = config_bamt.params["params"]
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
    # print(bn.get_info())

    # bn.plot(f'{title}_{mesh}_plot.html') # redefine

    # Parameters Learning and Sample
    bn.fit_parameters(df)

    # Sample() function
    df_res = df

    equations = []
    synth_data = bn.sample(config_bamt.params["glob_bamt"]["sample_k"], as_df=True)
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

    # save result
    if not (os.path.exists(f'data/{title}/bamt_result')):
        os.mkdir(f'data/{title}/bamt_result')

    number_of_files = len(os.listdir(path=f"data/{title}/bamt_result/"))

    if os.path.exists(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.csv'):
        with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}_{number_of_files}.pickle', 'wb') as f:
            pickle.dump(equations_result, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.pickle', 'wb') as f:
            pickle.dump(equations_result, f, pickle.HIGHEST_PROTOCOL)

    # # Load data
    # with open(f'data/{title}/bamt_result/data_equations_{config_bamt.params["glob_bamt"]["sample_k"]}.pickle', 'rb') as f:
    #     equations_result = pickle.load(f)

    return equations_result
