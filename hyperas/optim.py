from hyperopt import fmin
from .ensemble import VotingModel
import os
import inspect
import re
import sys
sys.path.append(".")


def minimize(model, data, algo, max_evals, trials, rseed=1337):
    """Minimize a keras model for given data and implicit hyperparameters.

    Parameters
    ----------
    model: A function defining a keras model with hyperas templates, which returns a
        valid hyperopt results dictionary, e.g.
        return {'loss': -acc, 'status': STATUS_OK}
    data: A parameterless function that defines and return all data needed in the above
        model definition.
    algo: A hyperopt algorithm, like tpe.suggest or rand.suggest
    max_evals: Maximum number of optimization runs
    trials: A hyperopt trials object, used to store intermediate results for all
        optimization runs
    rseed: Integer random seed for experiments

    Returns
    -------
    A pair consisting of the results dictionary of the best run and the corresponing
    keras model.
    """
    best_run = base_minimizer(model, data, algo, max_evals, trials, rseed)

    best_model = None
    for trial in trials:
        vals = trial.get('misc').get('vals')
        for key in vals.keys():
            vals[key] = vals[key][0]
        if trial.get('misc').get('vals') == best_run and 'model' in trial.get('result').keys():
            best_model = trial.get('result').get('model')

    return best_run, best_model


def best_ensemble(nb_ensemble_models, model, data, algo, max_evals, trials, voting='hard', weights=None):
    model_list = best_models(nb_models=nb_ensemble_models, model=model,
                             data=data, algo=algo, max_evals=max_evals, trials=trials)
    return VotingModel(model_list, voting, weights)


def best_models(nb_models, model, data, algo, max_evals, trials):
    base_minimizer(model, data, algo, max_evals, trials)
    if len(trials) < nb_models:
        nb_models = len(trials)
    scores = [trial.get('result').get('loss') for trial in trials]
    cut_off = sorted(scores, reverse=True)[nb_models-1]
    model_list = [trial.get('result').get('model') for trial in trials if trial.get('result').get('loss') >= cut_off]
    return model_list


def get_hyperopt_model_string(model, data):
    model_string = inspect.getsource(model)
    lines = model_string.split("\n")
    lines = [line for line in lines if not line.strip().startswith('#')]

    calling_script_file = os.path.abspath(inspect.stack()[-1][1])
    with open(calling_script_file, 'r') as f:
        calling_lines = f.read().split('\n')
        raw_imports = [line.strip() + "\n" for line in calling_lines if "import" in line]
        imports = ''.join(raw_imports)

    model_string = [line + "\n" for line in lines if "import" not in line]
    model_string = ''.join(model_string)

    parts = hyperparameter_names(model_string)
    aug_parts = augmented_names(parts)

    hyperopt_params = get_hyperparameters(model_string)
    space = get_hyperopt_space(parts, hyperopt_params)

    data_string = retrieve_data_string(data)
    model = hyperopt_keras_model(model_string, parts, aug_parts)

    temp_str = temp_string(imports, model, data_string, space)
    return temp_str


def base_minimizer(model, data, algo, max_evals, trials, rseed=1337, full_model_string=None):

    if full_model_string is not None:
        model_str = full_model_string
    else:
        model_str = get_hyperopt_model_string(model, data)
    write_temp_files(model_str)

    try:
        from temp_model import keras_fmin_fnct, get_space
    except:
        print("Unexpected error: {}".format(sys.exc_info()[0]))
        raise
    try:
        os.remove('./temp_model.py')
        os.remove('./temp_model.pyc')
    except OSError:
        pass

    best_run = fmin(keras_fmin_fnct,
                    space=get_space(),
                    algo=algo,
                    max_evals=max_evals,
                    trials=trials,
                    rseed=rseed)

    return best_run


def get_hyperopt_space(parts, hyperopt_params):
    space = "def get_space():\n    return {\n"
    for name, param in zip(parts, hyperopt_params):
        param = re.sub(r"\(", "('" + name + "', ", param, 1)
        space += "        '" + name + "': hp." + param + ",\n"
    space = space[:-1]
    space += "\n    }\n"
    print('>>> Hyperas search space:\n')
    print(space)
    return space


def retrieve_data_string(data):
    '''
    This assumes 4 spaces for indentation and won't work otherwise
    '''
    data_string = inspect.getsource(data)
    first_line = data_string.split("\n")[0]
    data_string = data_string.replace(first_line, "")
    data_string = re.sub(r"return.*", "", data_string)

    split_data = data_string.split("\n")
    for i, line in enumerate(split_data):
        split_data[i] = line[4:] + "\n"
    data_string = ''.join(split_data)
    print(">>> Data")
    print(data_string)
    return data_string


def hyperparameter_names(model_string):
    parts = []
    params = re.findall(r"(\{\{[^}]+}\})", model_string)
    for param in params:
        name = re.findall(r"(\w+(?=\s*[\=\(]\s*" + re.escape(param) + r"))", model_string)
        if len(name) > 0:
            parts.append(name[0])
        else:
            parts.append(parts[-1])
    # parts = re.findall(r"(\w+(?=\s*[\=\(]\s*\{\{[^}]+}\}))", model_string)
    print("PARTS:")
    for part in parts:
        print(part)
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


def hyperopt_keras_model(model_string, parts, aug_parts):
    first_line = model_string.split("\n")[0]
    model_string = model_string.replace(first_line, "def keras_fmin_fnct(space):\n")
    result = re.sub(r"(\{\{[^}]+}\})", lambda match: aug_parts.pop(0), model_string, count=len(parts))
    print('>>> Resulting replaced keras model:\n')
    print(result)
    return result


def temp_string(imports, model, data, space):
    temp = (imports + "from hyperopt import fmin, tpe, hp, STATUS_OK, Trials\n" +
            "from hyperas.distributions import conditional\n" +
            data + model + "\n" + space)
    return temp


def write_temp_files(tmp_str, path='./temp_model.py'):
    with open(path, 'w') as f:
        f.write(tmp_str)
        f.close()
    return
