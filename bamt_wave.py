import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import bamt.Networks as Nets
import bamt.Nodes as Nodes
import bamt.Preprocessors as pp
from pgmpy.estimators import K2Score

from sklearn import preprocessing


def bs_experiment(df, sample_k, mesh, title):

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

    params = {'init_nodes': ['d^2u/dx2^2{power: 1.0}']}  # !!! params only for wave_equation
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
    # print(bn.get_info())

    bn.plot(f'{title}_{mesh}_plot.html')

    # Parameters Learning and Sample
    bn.fit_parameters(df)

    # Sample() function
    df_res = df

    equations = []
    synth_data = bn.sample(sample_k, as_df=True)
    for i in range(len(synth_data)):
        equation = {}
        for col in df_res.columns:
            equation[synth_data[col].name] = synth_data[col].values[i]
        equations.append(equation)

    equations_result = []
    for i in range(len(synth_data)):
        equation_res = {}
        for key, value in equations[i].items():
            if abs(float(value)) > 0.001:
                equation_res[key] = value
        equations_result.append(equation_res)
        print(f'{i + 1}.{equation_res}')

    return equations_result
