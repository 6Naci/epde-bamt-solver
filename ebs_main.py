import dill as pickle
import pandas as pd
import numpy as np
import torch
import os

from func import load_data
from epde_general import epde_equations
# from bamt_general import bs_experiment
# from solver_general import solver_equations
# from func import confidence_region as conf_plt

if __name__ == '__main__':

    tasks = {
        'wave_equation': load_data.wave_equation, # 0
        'burgers_equation': load_data.burgers_equation, # 1
        'burgers_equation_small_grid': load_data.burgers_equation_small_grid, # 2
        'KdV_equation': load_data.KdV_equation, # 3
        'hunter_prey': load_data.hunter_prey # 4
    }

    title = list(tasks.keys())[4]  # name of the problem/equation

    u, grid_u, derivatives, cfg, params, b_conds = tasks[title]()

    for variance in cfg.params["global_config"]["variance_arr"]:

        df_main = epde_equations(u, grid_u, derivatives, cfg, variance, title)

        # equations = bs_experiment(df_main, cfg, title)
        #
        # u_main, models = solver_equations(cfg, params, b_conds, equations, epde_search_obj, title)
        #
        # conf_plt.confidence_region_print(u, cfg, params, u_main, variance)
