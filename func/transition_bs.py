import re
import random


def solver_view(object_row, cfg):
    """
        Transition from the type of BAMT output data to the type required by SOLVER.

        Parameters
        ----------
        object_row : dict
        object is system - {'u{power: 1.0} * v{power: 1.0}_u': -1.305, 'du/dx1{power: 1.0}_u': -0.0505,
                            'v{power: 1.0} * u{power: 1.0}_v': 0.971, 'du/dx1{power: 1.0}_r_u': '-1',
                            'dv/dx1{power: 1.0}_r_v': '-1', 'C_u': 0.332, 'u{power: 1.0}_v': -0.062, 'C_v': 0.324}
        object is equation - {'u{power: 1.0} * d^2u/dx1^2{power: 1.0}_u': 0.0203, 'u{power: 1.0}_u': -0.604,
                            'C_u': 0.921, 'd^2u/dx2^2{power: 1.0}_r_u': '-1'}.
        object old version (only equation) - {'u{power: 1.0} * d^2u/dx1^2{power: 1.0}': 0.0203, 'u{power: 1.0}': -0.604,
                            'C': 0.921, 'd^2u/dx2^2{power: 1.0}_r': '-1'}.

        cfg : class Config from TEDEouS/config.py contains the initial configuration of the task

        Returns
        -------
        object_main : list of dictionaries all object (system/equation)
    """

    reverse = cfg.params["glob_solver"]["reverse"]
    variable_names = cfg.params["fit"]["variable_names"]

    object_list = [{} for i in variable_names]

    # splitting an object into separate dictionaries
    for term, coeff in object_row.items():
        for i, elem in enumerate(variable_names):
            if '_' + elem in term:
                object_list[i][term.replace('_' + elem, "")] = coeff

    # for old version without variable_names
    if len(object_list) == 1 and not len(object_list[0].items()):
        object_list = [object_row]

    object_main = []

    for equation in object_list:
        equation_temp = equation_view(equation, cfg, reverse)
        object_main.append(equation_temp)

    return object_main


def equation_view(equation, cfg, reverse):
    """
        Transition from the type of BAMT output data to the type required by SOLVER.

        Parameters
        ----------
        equation : dict
            equation in form {'u{power: 1.0} * d^2u/dx1^2{power: 1.0}': 0.02036869782557119,
            'u{power: 1.0}': -0.6043591746687335, 'C': 0.9219325066472699, 'd^2u/dx2^2{power: 1.0}_r': '-1'}.

        cfg : class Config from TEDEouS/config.py contains the initial configuration of the task

        reverse: bool param, where true - if the order of parameters EPDE, SOLVER is different, False otherwise

        Returns
        -------
        equation_main : dict
            equation_main = {
        'u{power: 1.0} * d^2u/dx1^2{power: 1.0}':
            {
                'coeff': 0.02036869782557119,
                'term_set': [[None], [0, 0]],
                'power_set': [1.0, 1.0]
            },
        'u{power: 1.0}':
            {
                'coeff': -0.6043591746687335,
                'term_set': [None],
                'power_set': 1.0
            },
        'C':
            {
                'coeff': 0.9219325066472699,
                'term_set': [None],
                'power_set': 0
                'var': random.randrange(len(variable_names))
            },
        'd^2u/dx2^2{power: 1.0}_r':
            {
                'coeff': -1.0,
                'term_set': [1, 1],
                'power_set': 1.0}
                    }
    """

    # initial params before fit-EPDE (global params)
    dimensionality = cfg.params["global_config"]["dimensionality"]  # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
    max_deriv_order = max(cfg.params["fit"]["max_deriv_order"])
    variable_names = cfg.params["fit"]["variable_names"]

    equation_main = {}
    unknown_variables = {}  # x1, x2, ..., xn
    for i in range(dimensionality + 1):
        if reverse:
            unknown_variables[f'x{i + 1}'] = dimensionality - i  # x1 = 1, x2 = 0, because (epde = [t, x], solver = [x, t])
        else:
            unknown_variables[f'x{i + 1}'] = i # x1 = 0, x2 = 1, because (epde = [x, t], solver = [x, t])

    for term_i, value_i in equation.items():

        arr_term = term_i.split('*')
        term_set, power_set, vars_set = [], [], []

        for token_i in arr_term:

            if token_i != 'C':  # for the free term without params_ranges
                token, params = re.split("{|}", token_i)[:-1]  # tokens_name separated from params_ranges
            else:
                term_set, token, params, vars_set = [[None]], token_i, '', random.randrange(len(variable_names))

            power = float(params[params.find(' '):]) if 'power' in params else 0  # for one param - power
            power_set.append(power)

            for count, elem in enumerate(variable_names):
                if elem in token_i:
                    vars_set.append(count)
                    deriv = dev_variable(token, elem, unknown_variables, max_deriv_order)
                    term_set.append(deriv)

        term_set = term_set[0] if len(term_set) == 1 else term_set
        power_set = power_set[0] if len(power_set) == 1 else power_set

        term_main = {'coeff': float(value_i), 'term_set': term_set, 'power_set': power_set}

        if len(variable_names) > 1: # only for systems
            term_main['vars_set'] = vars_set

        equation_main[f'{term_i}'] = term_main

    return equation_main


def dev_variable(term, elem, unknown_var, max_order):  #
    """
        For variable definition and write derivatives

        Parameters
        ----------
        term: part of the structure
        elem: param from variable_names - list of objective function names
        unknown_var: dict - {x1: 0, x2: 1, ... xn: n-1}
        max_order: -

        Returns
        -------
        deriv : list of derivatives for token
    """
    for key, value in unknown_var.items():
        if key in term:
            for n in range(max_order):
                token_d = f'd{elem}/d{key}' if not n else f'd^{n + 1}{elem}/d{key}^{n + 1}'
                if token_d in term:
                    return [value] * (n + 1)
    return [None]


# params_ranges = ['power'] # {'power': (1, 1), 'freq': (0.95, 1.05), 'dim': (0, dimensionality)} or ['power', 'freq', 'dim'] for sin/cos

# from object type Equation need to take params

# not updated (yet not general view)
def view_for_create_eq(equation):
    """
        Transition from the type of BAMT output data to the type required by create class Equation

        Parameters
        ----------
        equation : dict
            equation in form {'d^2u/dx1^2{power: 1.0}': 0.02036869782557119,
            'u{power: 1.0}': -0.6043591746687335, 'C': 0.9219325066472699, 'd^2u/dx2^2{power: 1.0}_r': '-1'}.

        Returns
        -------
        0.02036869782557119 * d^2u/dx2^2{power: 1.0} + -0.6043591746687335 * u{power: 1.0} + 0.9219325066472699 = d^2u/dx1^2{power: 1.0}
    """
    flag = any(['_r' in k for k in list(equation)])  # checking whether there is a right part
    if not flag:
        eq = equation.copy()
        if "C" in list(equation):
            del eq["C"]
        value_max = max(list(eq.values()), key=abs)
        term_max = list(equation.keys())[list(equation.values()).index(value_max)]
        for key, value in equation.items():
            equation[key] = value / (-value_max)
        equation[term_max + '_r'] = equation.pop(term_max)

    form_left, form_c, form_right = '', '', ''
    right_part = False

    for key, value in equation.items():
        if '_r' not in key or right_part:
            if 'C' not in key:
                form_left += str(value) + ' * ' + key + ' + '
            else:
                form_c += str(value)
        else:
            form_right += ' = ' + key[:-2]  # removing the '_r'
            right_part = True

        if 'C' not in equation.keys():
            form_c = str(0.)
    return form_left + form_c + form_right
