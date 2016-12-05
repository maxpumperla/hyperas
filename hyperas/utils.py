import ast
from operator import attrgetter
import re
import warnings


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


def extract_imports(source, verbose=True):
    tree = ast.parse(source)
    import_parser = ImportParser()
    import_parser.visit(tree)
    import_lines = []
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
    lines = [line for line in source.split('\n') if not line.strip().startswith('#')]
    lines_to_remove = set(import_parser.line_numbers)
    non_import_lines = [line for i, line in enumerate(lines, start=1) if i not in lines_to_remove]
    return '\n'.join(non_import_lines)


def remove_all_comments(source):
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)  # remove '''...''' comments
    string = re.sub(re.compile("#.*?\n"), "\n", string)  # remove #...\n comments
    return string


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
                          'Found "%s" (length: %i) as well as "%s" (length: %i)' % (indent, len(indent), new_indent, len(new_indent)))
        indent = new_indent
    return indent
