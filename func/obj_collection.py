import re
import numpy as np
import itertools
import pandas as pd

regex = re.compile('freq:\s\d\S\d+')  # Using regular expression for frequency delete (sin/cos)


def dict_update(d_main, term, coeff, k):

    str_t = '_r' if '_r' in term else ''
    arr_term = re.sub('_r', '', term).split(' * ')

    # if structure recorded b * a provided, that a * b already exists (for all case - generalization)
    perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
    structure_added = False

    for p_i in perm_set:
        temp = " * ".join([arr_term[i] for i in p_i]) + str_t
        if temp in d_main:
            d_main[temp] += [0 for i in range(k - len(d_main[temp]))] + [coeff]
            structure_added = True

    if not structure_added:
        d_main[term] = [0 for i in range(k)] + [coeff]

    return d_main


def equation_table(k, equation, dict_main, dict_right):
    """
    Collecting the obtained values (coefficients and structures) into a common table (the right and left parts of the equation are considered separately)

    Parameters
        ----------
        equation:
        k : Number of equations (final)
        dict_main : dict/table coefficients left parts of the equation
        dict_right : -//- the right parts of the equation

        Returns
        -------
        dict_main, dict_right, k
    """

    equation_s = equation.structure  # list of class objects 'epde.structure.Term'
    equation_c = equation.weights_final  # coefficients of the right part
    text_form_eq = regex.sub('', equation.text_form)  # full equation line

    flag = False  # flag of the right part
    for t_eq in equation_s:
        term = regex.sub('', t_eq.name)  # full name term
        for t in range(len(equation_c)):
            c = equation_c[t]
            if f'{c} * {term} +' in text_form_eq:
                dict_main = dict_update(dict_main, term, c, k)
                equation_c = np.delete(equation_c, t)
                break
            elif f'+ {c} =' in text_form_eq:
                dict_main = dict_update(dict_main, "C", c, k)
                equation_c = np.delete(equation_c, t)
                break
        if f'= {term}' == text_form_eq[text_form_eq.find('='):] and flag is False:
            flag = True
            term += '_r'
            dict_right = dict_update(dict_right, term, -1., k)

    return [dict_main, dict_right]


def object_table(res, variable_name, table_main, k):
    """
    Collecting the obtained objects (system/equation) into a common table

    Parameters
        ----------
        variable_name:
        res : Pareto front of detected equations/systems
        table_main: List of dictionaries
        k : Number of equations/system (final)

        Returns
        -------
        table_main: [{'variable_name1': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]},
                    {'variable_name2': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]}]
    """
    for list_SoEq in res:  # List SoEq - an object of the class 'epde.structure.main_structures.SoEq'
        for SoEq in list_SoEq:
            # variable_name = SoEq.vals.equation_keys # param for object_epde_search
            for n, value in enumerate(variable_name):
                gene = SoEq.vals.chromosome.get(value)
                table_main[n][value] = equation_table(k, gene.value, *table_main[n][value])

            k += 1
            print(k)

    return table_main, k


def update_data(df):
    # Rounding values
    for col in df.columns:
        df[col] = df[col].round(decimals=10)

    # Deleting null columns
    df = df.loc[:, (df != 0).any(axis=0)]

    df_new = df
    for col in df_new.columns:
        if '_r' not in col and col + "_r" in df_new.columns:  # union of repeated structures
            temp = df_new[col + "_r"] + df_new[col]
            arr_value = temp.unique()
            arr_value.sort()
            if len(arr_value) == 2 and (arr_value == np.array([-1, 0])).all():  # separation of the structures of the right part (for conversion to a discrete type)
                df_new[col + "_r"] = df_new[col + "_r"] + df_new[col]
                df_new = df_new.drop(col, axis=1)
            else:
                df_new[col] = df_new[col + "_r"] + df_new[col]
                df_new = df_new.drop(col + "_r", axis=1)

    return df_new


def preprocessing_bamt(variable_name, table_main, k):
    data_frame_total = pd.DataFrame()

    # connecting right/left parts of the equation in each dictionary
    for dict_var in table_main:
        for var_name, list_structure in dict_var.items(): # object - {'variable_name1': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]}
            general_dict = {}
            for structure in list_structure:
                general_dict.update(structure)
            dict_var[var_name] = general_dict

    # filling with zeros
    for dict_var in table_main:
        for var_name, general_dict in dict_var.items():
            for key, value in general_dict.items():  # value - it's general dictionary (list dict -> dict)
                if len(value) < k:
                    general_dict[key] = general_dict[key] + [0. for i in range(k - len(general_dict[key]))]

    data_frame_main = [{i: pd.DataFrame()} for i in variable_name]
    # creating dataframe from a table and updating the data
    for n, dict_var in enumerate(table_main):
        for var_name, general_dict in dict_var.items():
            temp = pd.DataFrame(general_dict)
            total_result = update_data(temp)
            data_frame_main[n][var_name] = total_result

    for n, dict_var in enumerate(variable_name):
        data_frame_temp = data_frame_main[n].get(dict_var).copy()
        # renaming columns for every dataframe (column_{variable_name})
        list_columns = [f'{i}_{dict_var}' for i in data_frame_temp.columns]
        data_frame_temp.columns = list_columns
        # combine dataframes
        data_frame_total = pd.concat([data_frame_total, data_frame_temp], axis=1)

    return data_frame_total
