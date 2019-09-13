import ast
import re
import warnings
from operator import attrgetter

from hyperopt import space_eval


class ImportParser(ast.NodeVisitor):
    def __init__(self):
        self.lines = []
        self.line_numbers = []

    def visit_Import(self, node):
        line = 'import {}'.format(self._import_names(node.names))
        if (self._import_asnames(node.names) != ''):
            line += ' as {}'.format(self._import_asnames(node.names))
        self.line_numbers.append(node.lineno)
        self.lines.append(line)

    def visit_ImportFrom(self, node):
        line = 'from {}{} import {}'.format(
            node.level * '.',
            node.module or '',
            self._import_names(node.names))
        if (self._import_asnames(node.names) != ''):
            line += " as {}".format(self._import_asnames(node.names))
        self.line_numbers.append(node.lineno)
        self.lines.append(line)

    def _import_names(self, names):
        return ', '.join(map(attrgetter('name'), names))

    def _import_asnames(self, names):
        asname = map(attrgetter('asname'), names)
        return ''.join(filter(None, asname))


def extract_imports(source, verbose=True):
    tree = ast.parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    import_lines = ['#coding=utf-8\n']
    for line in import_parser.lines:
        if 'print_function' in line:
            import_lines.append(line + '\n')
        # skip imports for pycharm and eclipse
        elif '_pydev_' in line or 'java.lang' in line:
            continue
        else:
            import_lines.append('try:\n    {}\nexcept:\n    pass\n'.format(line))
    imports_str = '\n'.join(import_lines)
    if verbose:
        print('>>> Imports:')
        print(imports_str)
    return imports_str


def remove_imports(source):
    tree = ast.parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    lines = source.split('\n')  # the source including all comments, since we parse the line numbers with comments!
    lines_to_remove = set(import_parser.line_numbers)
    non_import_lines = [line for i, line in enumerate(lines, start=1) if i not in lines_to_remove]
    return '\n'.join(non_import_lines)


def remove_all_comments(source):
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)  # remove '''...''' comments
    string = re.sub(re.compile("(?<!('|\").)*#[^'\"]*?\n"), "\n", string)  # remove #...\n comments
    return string


def temp_string(imports, model, data, functions, space):
    temp = (imports + "from hyperopt import fmin, tpe, hp, STATUS_OK, Trials\n" +
            functions + data + model + "\n" + space)
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
    code : string
       any multiline text, such as as (fragments) of source code

    Returns
    -------
    str : string
       The input with added <n>: for each line

    Example
    -------
    code = "def do_stuff(x):\n\tprint(x)\n"
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
                          'Found "%s" (length: %i) as well as "%s" (length: %i)' % (
                              indent, len(indent), new_indent, len(new_indent)))
        indent = new_indent
    return indent


def unpack_hyperopt_vals(vals):
    """
    Unpack values from a hyperopt return dictionary where values are wrapped in a list.
    :param vals: dict
    :return: dict
        copy of the dictionary with unpacked values
    """
    assert isinstance(vals, dict), "Parameter must be given as dict."
    ret = {}
    for k, v in list(vals.items()):
        try:
            ret[k] = v[0]
        except (TypeError, IndexError):
            ret[k] = v
    return ret


def eval_hyperopt_space(space, vals):
    """
    Evaluate a set of parameter values within the hyperopt space.
    Optionally unpacks the values, if they are wrapped in lists.
    :param space: dict
        the hyperopt space dictionary
    :param vals: dict
        the values from a hyperopt trial
    :return: evaluated space
    """
    unpacked_vals = unpack_hyperopt_vals(vals)
    return space_eval(space, unpacked_vals)


def find_signature_end(model_string):
    """
    Find the index of the colon in the function signature.
    :param model_string: string
        source code of the model
    :return: int
        the index of the colon
    """
    index, brace_depth = 0, 0
    while index < len(model_string):
        ch = model_string[index]
        if brace_depth == 0 and ch == ':':
            break
        if ch == '#':  # Ignore comments
            index += 1
            while index < len(model_string) and model_string[index] != '\n':
                index += 1
            index += 1
        elif ch in ['"', "'"]:  # Skip strings
            string_depth = 0
            while index < len(model_string) and model_string[index] == ch:
                string_depth += 1
                index += 1
            if string_depth == 2:
                string_depth = 1
            index += string_depth
            while index < len(model_string):
                if model_string[index] == '\\':
                    index += 2
                elif model_string[index] == ch:
                    string_depth -= 1
                    if string_depth == 0:
                        break
                    index += 1
                else:
                    index += 1
            index += 1
        elif ch == '(':
            brace_depth += 1
            index += 1
        elif ch == ')':
            brace_depth -= 1
            index += 1
        else:
            index += 1
    return index
