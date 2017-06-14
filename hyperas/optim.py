import inspect
import os
import re
import sys

import nbformat
import numpy as np
from hyperopt import fmin
from nbconvert import PythonExporter

from .ensemble import VotingModel
from .utils import (
    remove_imports, remove_all_comments, extract_imports, temp_string,
    write_temp_files, determine_indent, with_line_numbers, unpack_hyperopt_vals,
    eval_hyperopt_space)

sys.path.append(".")


def minimize(model,
             data,
             algo,
             max_evals,
             trials,
             functions=None,
             rseed=1337,
             notebook_name=None,
             verbose=True,
             eval_space=False,
             return_space=False):
    """
    Minimize a keras model for given data and implicit hyperparameters.

    Parameters
    ----------
    model: A function defining a keras model with hyperas templates, which returns a
        valid hyperopt results dictionary, e.g.
        return {'loss': -acc, 'status': STATUS_OK}
    data: A parameter-less function that defines and return all data needed in the above
        model definition.
    algo: A hyperopt algorithm, like tpe.suggest or rand.suggest
    max_evals: Maximum number of optimization runs
    trials: A hyperopt trials object, used to store intermediate results for all
        optimization runs
    rseed: Integer random seed for experiments
    notebook_name: If running from an ipython notebook, provide filename (not path)
    verbose: Print verbose output
    eval_space: Evaluate the best run in the search space such that 'choice's contain actually meaningful values instead
                of mere indices
    return_space: Return the hyperopt search space object (e.g. for further processing) as last return value

    Returns
    -------
    If `return_space` is False: A pair consisting of the results dictionary of the best run and the corresponding
    keras model.
    If `return_space` is True: The pair of best result and corresponding keras model, and the hyperopt search space
    """
    best_run, space = base_minimizer(model=model,
                                     data=data,
                                     functions=functions,
                                     algo=algo,
                                     max_evals=max_evals,
                                     trials=trials,
                                     rseed=rseed,
                                     full_model_string=None,
                                     notebook_name=notebook_name,
                                     verbose=verbose)

    best_model = None
    for trial in trials:
        vals = trial.get('misc').get('vals')
        # unpack the values from lists without overwriting the mutable dict within 'trial'
        unpacked_vals = unpack_hyperopt_vals(vals)
        # identify the best_run (comes with unpacked values from the hyperopt function `base.Trials.argmin`)
        if unpacked_vals == best_run and 'model' in trial.get('result').keys():
            best_model = trial.get('result').get('model')

    if eval_space is True:
        # evaluate the search space
        best_run = eval_hyperopt_space(space, best_run)

    if return_space is True:
        # return the space as well
        return best_run, best_model, space
    else:
        # the default case for backwards compatibility with expanded return arguments
        return best_run, best_model


def base_minimizer(model, data, functions, algo, max_evals, trials,
                   rseed=1337, full_model_string=None, notebook_name=None,
                   verbose=True, stack=3):
    if full_model_string is not None:
        model_str = full_model_string
    else:
        model_str = get_hyperopt_model_string(model, data, functions, notebook_name, verbose, stack)
    temp_file = './temp_model.py'
    write_temp_files(model_str, temp_file)

    try:
        from temp_model import keras_fmin_fnct, get_space
    except:
        print("Unexpected error: {}".format(sys.exc_info()[0]))
        raise
    try:
        os.remove(temp_file)
        os.remove(temp_file + 'c')
    except OSError:
        pass

    try:
        # for backward compatibility.
        return (
            fmin(keras_fmin_fnct,
                 space=get_space(),
                 algo=algo,
                 max_evals=max_evals,
                 trials=trials,
                 rseed=rseed,
                 return_argmin=True),
            get_space()
        )
    except TypeError:
        pass

    return (
        fmin(keras_fmin_fnct,
             space=get_space(),
             algo=algo,
             max_evals=max_evals,
             trials=trials,
             rstate=np.random.RandomState(rseed),
             return_argmin=True),
        get_space()
    )


def best_ensemble(nb_ensemble_models, model, data, algo, max_evals,
                  trials, voting='hard', weights=None, nb_classes=None, functions=None):
    model_list = best_models(nb_models=nb_ensemble_models,
                             model=model,
                             data=data,
                             algo=algo,
                             max_evals=max_evals,
                             trials=trials,
                             functions=functions)
    return VotingModel(model_list, voting, weights, nb_classes)


def best_models(nb_models, model, data, algo, max_evals, trials, functions=None):
    base_minimizer(model=model,
                   data=data,
                   functions=functions,
                   algo=algo,
                   max_evals=max_evals,
                   trials=trials,
                   stack=4)
    if len(trials) < nb_models:
        nb_models = len(trials)
    scores = [trial.get('result').get('loss') for trial in trials]
    cut_off = sorted(scores, reverse=True)[nb_models - 1]
    model_list = [trial.get('result').get('model') for trial in trials if trial.get('result').get('loss') >= cut_off]
    return model_list


def get_hyperopt_model_string(model, data, functions, notebook_name, verbose, stack):
    model_string = inspect.getsource(model)
    model_string = remove_imports(model_string)

    if notebook_name:
        notebook_path = os.getcwd() + "/{}.ipynb".format(notebook_name)
        with open(notebook_path, 'r') as f:
            notebook = nbformat.reads(f.read(), nbformat.NO_CONVERT)
            exporter = PythonExporter()
            source, _ = exporter.from_notebook_node(notebook)
    else:
        calling_script_file = os.path.abspath(inspect.stack()[stack][1])
        with open(calling_script_file, 'r') as f:
            source = f.read()

    cleaned_source = remove_all_comments(source)
    imports = extract_imports(cleaned_source, verbose)

    parts = hyperparameter_names(model_string)
    aug_parts = augmented_names(parts)

    hyperopt_params = get_hyperparameters(model_string)
    space = get_hyperopt_space(parts, hyperopt_params, verbose)

    functions_string = retrieve_function_string(functions, verbose)
    data_string = retrieve_data_string(data, verbose)
    model = hyperopt_keras_model(model_string, parts, aug_parts, verbose)

    temp_str = temp_string(imports, model, data_string, functions_string, space)
    return temp_str


def get_hyperopt_space(parts, hyperopt_params, verbose=True):
    space = "def get_space():\n    return {\n"
    for name, param in zip(parts, hyperopt_params):
        param = re.sub(r"\(", "('" + name + "', ", param, 1)
        space += "        '" + name + "': hp." + param + ",\n"
    space = space[:-1]
    space += "\n    }\n"
    if verbose:
        print('>>> Hyperas search space:\n')
        print(space)
    return space


def retrieve_data_string(data, verbose=True):
    data_string = inspect.getsource(data)
    first_line = data_string.split("\n")[0]
    indent_length = len(determine_indent(data_string))
    data_string = data_string.replace(first_line, "")
    r = re.compile(r'^\s*return.*')
    last_line = [s for s in reversed(data_string.split("\n")) if r.match(s)][0]
    data_string = data_string.replace(last_line, "")

    split_data = data_string.split("\n")
    for i, line in enumerate(split_data):
        split_data[i] = line[indent_length:] + "\n"
    data_string = ''.join(split_data)
    if verbose:
        print(">>> Data")
        print(with_line_numbers(data_string))
    return data_string


def retrieve_function_string(functions, verbose=True):
    function_strings = ''
    if functions is None:
        return function_strings
    for function in functions:
        function_string = inspect.getsource(function)
        function_strings = function_strings + function_string + '\n'
    if verbose:
        print(">>> Functions")
        print(with_line_numbers(function_strings))
    return function_strings


def hyperparameter_names(model_string):
    parts = []
    params = re.findall(r"(\{\{[^}]+}\})", model_string)
    for param in params:
        name = re.findall(r"(\w+(?=\s*[\=\(]\s*" + re.escape(param) + r"))", model_string)
        if len(name) > 0:
            parts.append(name[0])
        else:
            parts.append(parts[-1])
    part_dict = {}
    for i, part in enumerate(parts):
        if part in part_dict.keys():
            part_dict[part] += 1
            parts[i] = part + "_" + str(part_dict[part])
        else:
            part_dict[part] = 0
    return parts


def get_hyperparameters(model_string):
    hyperopt_params = re.findall(r"(\{\{[^}]+}\})", model_string)
    for i, param in enumerate(hyperopt_params):
        hyperopt_params[i] = re.sub(r"[\{\}]", '', param)
    return hyperopt_params


def augmented_names(parts):
    aug_parts = []
    for i, part in enumerate(parts):
        aug_parts.append("space['" + part + "']")
    return aug_parts


def hyperopt_keras_model(model_string, parts, aug_parts, verbose=True):
    first_line = model_string.split("\n")[0]
    model_string = model_string.replace(first_line, "def keras_fmin_fnct(space):\n")
    result = re.sub(r"(\{\{[^}]+}\})", lambda match: aug_parts.pop(0), model_string, count=len(parts))
    if verbose:
        print('>>> Resulting replaced keras model:\n')
        print(with_line_numbers(result))
    return result
