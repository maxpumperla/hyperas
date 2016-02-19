from hyperopt import fmin
import os


def minimize(fitted_model, algo, max_evals, trials):
    import inspect
    import re
    model_string = inspect.getsource(fitted_model)

    parts = re.findall(r"(\w+(?=\s*[\=\(]\s*\{\{[^}]+}\}))", model_string)
    part_dict = {}
    for i, part in enumerate(parts):
        if part in part_dict.keys():
            part_dict[part] += 1
            parts[i] = part + "_" + str(part_dict[part])
        else:
            part_dict[part] = 0

    aug_parts = []
    for (i, part) in enumerate(parts):
        aug_parts.append("space['" + part + "']")

    hyperopt_params = re.findall(r"(\{\{[^}]+}\})", model_string)
    for (i, param) in enumerate(hyperopt_params):
        hyperopt_params[i] = re.sub(r"[\{\}]", '', param)

    space = "def get_space(): \n return {\n"
    for name, param in zip(parts, hyperopt_params):
        param = re.sub(r"\(", "('" + name + "', ", param)
        space += "'" + name + "': hp." + param + ","
    space = space[:-1]
    space += "\n}"
    print(space)

    first_line = model_string.split("\n")[0]
    model_string = model_string.replace(first_line, "def keras_fmin_fnct(space):\n")
    result = re.sub(r"(\{\{[^}]+}\})", lambda match: aug_parts.pop(0), model_string, count=len(parts))

    with open('./temp.py', 'w') as f:
        f.write("from hyperopt import fmin, tpe, hp, STATUS_OK, Trials\n")
        f.write(result)
        f.write("\n\n")
        f.write(space)
        f.close()

    import sys
    sys.path.append(".")
    from temp import keras_fmin_fnct, get_space
    os.remove('./temp.py')
    os.remove('./temp.pyc')
    sys.path.remove(".")

    best = fmin(keras_fmin_fnct, space=get_space(), algo=algo, max_evals=max_evals, trials=trials)
    return best
