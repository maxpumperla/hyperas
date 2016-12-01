import numpy as np
from hyperopt import fmin
from .ensemble import VotingModel
import ast
import inspect
from operator import attrgetter
import os
import re
import sys
import warnings

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


def best_ensemble(nb_ensemble_models, model, data, algo, max_evals, trials, voting='hard', weights=None, nb_classes=None):
    model_list = best_models(nb_models=nb_ensemble_models, model=model,
                             data=data, algo=algo, max_evals=max_evals, trials=trials)
    return VotingModel(model_list, voting, weights, nb_classes)


def best_models(nb_models, model, data, algo, max_evals, trials):
    base_minimizer(model, data, algo, max_evals, trials)
    if len(trials) < nb_models:
        nb_models = len(trials)
    scores = [trial.get('result').get('loss') for trial in trials]
    cut_off = sorted(scores, reverse=True)[nb_models - 1]
    model_list = [trial.get('result').get('model') for trial in trials if trial.get('result').get('loss') >= cut_off]
    return model_list


class ImportParser(ast.NodeVisitor):

    def __init__(self):
        self.lines = []
        self.line_numbers = []

    def visit_Import(self, node):
        line = 'import {}'.format(self._import_names(node.names))
        self.line_numbers.append(node.lineno)
        self.lines.append(line)

    def visit_ImportFrom(self, node):
        line = 'from {}{} import {}'.format(
            node.level * '.',
            node.module or '',
            self._import_names(node.names))
        self.line_numbers.append(node.lineno)
        self.lines.append(line)

    def _import_names(self, names):
        return ', '.join(map(attrgetter('name'), names))


def extract_imports(source):
    tree = ast.parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    import_lines = []
    for line in import_parser.lines:
        if 'print_function' in line:
            import_lines.append(line + '\n')
        elif '_pydev_' in line:
            continue
        else:
            import_lines.append('try:\n    {}\nexcept:\n    pass\n'.format(line))
    imports_str = '\n'.join(import_lines)
    return imports_str


def remove_imports(source):
    tree = ast.parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    lines = [line for line in source.split('\n') if not line.strip().startswith('#')]
    lines_to_remove = set(import_parser.line_numbers)
    non_import_lines = [line for i, line in enumerate(lines, start=1) if i not in lines_to_remove]
    return '\n'.join(non_import_lines)


def remove_all_comments(source):
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)  # remove '''...''' comments
    string = re.sub(re.compile("#.*?\n"), "", string)  # remove #...\n comments
    return string


def get_hyperopt_model_string(model, data):
    model_string = inspect.getsource(model)
    model_string = remove_imports(model_string)

    calling_script_file = os.path.abspath(inspect.stack()[-1][1])
    with open(calling_script_file, 'r') as f:
        source = f.read()
        cleaned_source = remove_all_comments(source)
        imports = extract_imports(cleaned_source)

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

    try:  # for backward compatibility.
        best_run = fmin(keras_fmin_fnct,
                        space=get_space(),
                        algo=algo,
                        max_evals=max_evals,
                        trials=trials,
                        rseed=rseed)
    except TypeError:
        best_run = fmin(keras_fmin_fnct,
                        space=get_space(),
                        algo=algo,
                        max_evals=max_evals,
                        trials=trials,
                        rstate=np.random.RandomState(rseed))

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
    data_string = inspect.getsource(data)
    first_line = data_string.split("\n")[0]
    indent_length = len(determine_indent(data_string))
    data_string = data_string.replace(first_line, "")
    data_string = re.sub(r"return.*", "", data_string)

    split_data = data_string.split("\n")
    for i, line in enumerate(split_data):
        split_data[i] = line[indent_length:] + "\n"
    data_string = ''.join(split_data)
    print(">>> Data")
    print(with_line_numbers(data_string))
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
    print(with_line_numbers(result))
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


def with_line_numbers(code):
    """
    Adds line numbers to each line of a source code fragment

    Parameters
    ----------
    str : string
       any multiline text, such as as (fragments) of source code

    Returns
    -------
    str : string
       The input with added <n>: for each line

    Example
    -------
    code = "def do_stuff(x)\n\n    print(x)\n"
    with_line_numbers(code)

    1: def do_stuff(x):
    2:     print(x)
    3:
    """
    max_number_length = str(len(str(len(code))))
    format_str = "{:>" + max_number_length + "d}: {:}"
    return "\n".join([format_str.format(line_number + 1, line) for line_number, line in enumerate(code.split("\n"))])


def determine_indent(str):
    """
    Figure out the character(s) used for indents in a given source code fragement.

    Parameters
    ----------
    str : string
      source code starting at an indent of 0 and containing at least one indented block.

    Returns
    -------
    string
      The character(s) used for indenting.

    Example
    -------
    code = "def do_stuff(x)\n   print(x)\n"
    indent = determine_indent(str)
    print("The code '", code, "' is indented with \n'", indent, "' (size: ", len(indent), ")")
    """
    indent = None
    reg = r"""
      ^(?P<previous_indent>\s*)\S.+?:\n      # line starting a block, i. e. '   for i in x:\n'
      ((\s*)\n)*                             # empty lines
      (?P=previous_indent)(?P<indent>\s+)\S  # first indented line of the new block, i. e. '      d'(..oStuff())
      """

    matches = re.compile(reg, re.MULTILINE | re.VERBOSE).finditer(str)
    for block_start in matches:
        new_indent = block_start.groupdict()['indent']
        if indent and new_indent != indent:
            warnings.warn('Inconsistent indentation detected.'
                          'Found "%s" (length: %i) as well as "%s" (length: %i)' % (indent, len(indent), new_indent, len(new_indent)))
        indent = new_indent
    return indent
