DEFAULT_CONFIG_EBS = '''
{
"discovery_module": {
"name_module": "EPDE"
},

"SINDY_config": "commentary",
"PDELibrary": {
"derivative_order": 3,
"include_bias": true,
"is_uniform": true,
"include_interaction":true
},
"set_optimizer": {
"type": "SR3"
},
"STLSQ": {
"threshold": 5,
"alpha": 1e-5,
"normalize_columns": true
},
"SR3": {
"threshold": 7,
"max_iter": 1000,
"tol": 1e-15,
"nu": 1e2,
"thresholder": "l0",
"normalize_columns": true
},

"EPDE_config": "commentary",
"epde_search": {
"use_solver": false,
"eq_search_iter": 100
},
"set_memory_properties": {
"mem_for_cache_frac": 10
},
"set_moeadd_params": {
"population_size": 10,
"training_epochs": 5
},
"Cache_stored_tokens": {
"token_type": "grid",
"token_labels": ["t", "x"],
"params_ranges": {"power": [1, 1]},
"params_equality_ranges": null
},
"fit": {
"max_deriv_order": [2, 2],
"boundary": 0,
"equation_terms_max_number": 3,
"equation_factors_max_number": 1,
"eq_sparsity_interval": [1e-8, 5.0],
"derivs": null,
"deriv_method": "poly",
"deriv_method_kwargs": {"smooth": true},
"memory_for_cache": 25,
"prune_domain": false
},
"glob_epde": {
"test_iter_limit": 1,
"variance_arr": [0],
"save_result": true,
"load_result": false
},

"BAMT_config": "commentary",
"glob_bamt": {
"sample_k": 35,
"lambda": 0.001,
"plot": false,
"save_equations": true,
"load_equations": false
},
"params": {
"init_nodes": false
},

"SOLVER_config": "commentary",
"glob_solver": {
"mode": "NN"
},
"Optimizer": {
"learning_rate":1e-4,
"lambda_bound":10,
"optimizer":"Adam"
},
"Cache":{
"use_cache":true,
"cache_dir":"../cache/",
"cache_verbose":false,
"save_always":false,
"model_randomize_parameter":0
},
"NN":{
"batch_size":null,
"lp_par":null,
"grid_point_subset":["central"],
"h":0.001
},
"Verbose":{
"verbose":true,
"print_every":null
},
"StopCriterion":{
"eps":1e-5,
"tmin":1000,
"tmax":1e5 ,
"patience":5,
"loss_oscillation_window":100,
"no_improvement_patience":1000   	
},
"Matrix":{
"lp_par":null,
"cache_model":null
}
}
'''