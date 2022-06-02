import re


def dev_variable(term, unknown_var, max_order):  # for variable definition and write derivatives
    # for n in range(max_order):
    #     if 'du' in term or f'd^{n + 1}u' in term: # found out the order
    #         print(n + 1)

    for key, value in unknown_var.items():
        if key in term:
            for n in range(max_order):
                # if mixed partial derivative ? d^2u/dx1dx2 (how a separate token)
                token_d = f'du/d{key}' if not n else f'd^{n + 1}u/d{key}^{n + 1}'
                if token_d in term:
                    return [value] * (n + 1)
    return [None]


def solver_view(equation):
    """
        Transition from the type of BAMT output data to the type required by SOLVER.

        Parameters
        ----------
        equation : dict
            equation in form {'u{power: 1.0} * d^2u/dx1^2{power: 1.0}': 0.02036869782557119,
            'u{power: 1.0}': -0.6043591746687335, 'C': 0.9219325066472699, 'd^2u/dx2^2{power: 1.0}_r': '-1'}.

        Returns
        -------
        equation_main : dict
            equation_main = {
        'u{power: 1.0} * d^2u/dx1^2{power: 1.0}':
            {
                'coeff': 0.02036869782557119,
                'vars_set': [[None], [0, 0]],
                'power_set': [1.0, 1.0]
            },
        'u{power: 1.0}':
            {
                'coeff': -0.6043591746687335,
                'vars_set': [None],
                'power_set': 1.0
            },
        'C':
            {
                'coeff': 0.9219325066472699,
                'vars_set': [None],
                'power_set': 0
            },
        'd^2u/dx2^2{power: 1.0}_r':
            {
                'coeff': -1.0,
                'vars_set': [1, 1],
                'power_set': 1.0}
                    }
    """

    # initial params before fit-EPDE (global params)
    dimensionality = 2
    max_deriv_order = 2

    equation_main = {}
    unknown_variables = {}  # x1, x2, ..., xn
    for i in range(dimensionality):
        unknown_variables[
            f'x{i + 1}'] = dimensionality - 1 - i  # x1 = 1, x2 = 0, because (epde = [t, x], solver = [x, t])

    for term_i, value_i in equation.items():

        arr_term = term_i.split('*')
        vars_set, power_set = [], []

        for token_i in arr_term:

            if token_i != 'C':  # for the free term without params_ranges
                token, params = re.split("{|}", token_i)[:-1]  # tokens_name separated from params_ranges
            else:
                token, params = token_i, ''

            deriv = dev_variable(token, unknown_variables, max_deriv_order)
            vars_set.append(deriv)

            power = float(params[params.find(' '):]) if 'power' in params else 0  # for one param - power
            power_set.append(power)

        vars_set = vars_set[0] if len(vars_set) == 1 else vars_set
        power_set = power_set[0] if len(power_set) == 1 else power_set
        term_main = {'coeff': float(value_i), 'vars_set': vars_set, 'power_set': power_set}

        equation_main[f'{term_i}'] = term_main

    return equation_main


# params_ranges = ['power'] # {'power': (1, 1), 'freq': (0.95, 1.05), 'dim': (0, dimensionality)} or ['power', 'freq', 'dim'] for sin/cos

# from object type Equation need to take params


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
    form_left, form_c, form_right = '', '', ''

    for key, value in equation.items():
        if '_r' not in key:
            if 'C' not in key:
                form_left += str(value) + ' * ' + key + ' + '
            else:
                form_c += str(value)
        else:
            form_right += ' = ' + key
    return form_left + form_c + form_right[:-2]
