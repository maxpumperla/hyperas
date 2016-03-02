from hyperopt import fmin
import os
import inspect
import re
import sys
sys.path.append(".")


def minimize(model, data, algo, max_evals, trials):
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

    Returns
    -------
    A pair consisting of the results dictionary of the best run and the corresponing
    keras model.
    """
    model_string = inspect.getsource(model)
    lines = model_string.split("\n")
    raw_imports = [line.strip() + "\n" for line in lines if "import" in line]
    imports = ''.join(raw_imports)

    model_string = [line + "\n" for line in lines if "import" not in line]
    model_string = ''.join(model_string)

    parts = hyperparameter_names(model_string)
    aug_parts = augmented_names(parts)

    hyperopt_params = get_hyperparameters(model_string)
    space = get_hyperopt_space(parts, hyperopt_params)

    data_string = retrieve_data_string(data)
    model = hyperopt_keras_model(model_string, parts, aug_parts)

    write_temp_files(imports, model, data_string, space)

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
                    trials=trials)

    best_model = None
    for trial in trials:
        vals = trial.get('misc').get('vals')
        for key in vals.keys():
            vals[key] = vals[key][0]
        if trial.get('misc').get('vals') == best_run and 'model' in trial.get('result').keys():
            best_model = trial.get('result').get('model')

    return best_run, best_model


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
    data_string = inspect.getsource(data)
    first_line = data_string.split("\n")[0]
    data_string = data_string.replace(first_line, "")
    data_string = re.sub(r"return.*", "", data_string)

    split_data = data_string.split("\n")
    for i, line in enumerate(split_data):
        split_data[i] = line.strip() + "\n"
    data_string = ''.join(split_data)
    print(">>> Data")
    print(data_string)
    return data_string


def hyperparameter_names(model_string):
    parts = re.findall(r"(\w+(?=\s*[\=\(]\s*\{\{[^}]+}\}))", model_string)
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
    for (i, param) in enumerate(hyperopt_params):
        hyperopt_params[i] = re.sub(r"[\{\}]", '', param)
    return hyperopt_params


def augmented_names(parts):
    aug_parts = []
    for (i, part) in enumerate(parts):
        aug_parts.append("space['" + part + "']")
    return aug_parts


def hyperopt_keras_model(model_string, parts, aug_parts):
    first_line = model_string.split("\n")[0]
    model_string = model_string.replace(first_line, "def keras_fmin_fnct(space):\n")
    result = re.sub(r"(\{\{[^}]+}\})", lambda match: aug_parts.pop(0), model_string, count=len(parts))
    print('>>> Resulting replaced keras model:\n')
    print(result)
    return result


def write_temp_files(imports, model, data, space, path='./temp_model.py'):
    with open(path, 'w') as f:
        f.write("from hyperopt import fmin, tpe, hp, STATUS_OK, Trials\n")
        f.write("from hyperas.distributions import conditional\n")
        f.write(imports)
        f.write(data)
        f.write(model)
        f.write("\n")
        f.write(space)
        f.close()
    return
