import os
from hyperopt import hp
from hyperas.utils import (
    extract_imports, remove_imports, remove_all_comments, temp_string,
    write_temp_files, with_line_numbers, determine_indent, unpack_hyperopt_vals,
    eval_hyperopt_space, find_signature_end)

TEST_SOURCE = """
from __future__ import print_function
from sys import path
from os import walk as walk2
import os
import sys # ignore this comment
''' remove me '''
# import nocomment
from java.lang import stuff
from _pydev_ import stuff
from os.path import splitext as split
import os.path.splitext as sp
"""

TEST_SOURCE_2 = """
import sys
foo_bar()
"""

TEST_SOURCE_3 = """
def foo():
    # a comment in a function
    import sys
    bar()
"""

TEST_SOURCE_4 = """
@foo_bar(bar_foo)
def foo(train_x=')\\':', train_y=")\\":",  # ):
        test_x=lambda x: bar, test_y=bar[:, 0],
        foo='''
  ):):  
\\'''', bar="") :
    pass
"""


def test_extract_imports():
    result = extract_imports(TEST_SOURCE)
    assert 'java.lang' not in result
    assert 'nocomment' not in result
    assert '_pydev_' not in result
    assert 'try:\n    import os\nexcept:\n    pass\n' in result
    assert 'from sys import path' in result
    assert 'from os import walk as walk2' in result
    assert 'ignore' not in result
    assert 'remove me' not in result
    assert 'from __future__ import print_function' in result
    assert 'from os.path import splitext as split' in result
    assert 'import os.path.splitext as sp' in result


def test_remove_imports():
    result = remove_imports(TEST_SOURCE_2)
    assert 'foo_bar()' in result


def test_remove_imports_in_function():
    result = remove_imports(TEST_SOURCE_3)
    # test function should have 3 lines (including the comment)
    assert len(result.split('\n')[1:-1]) == 3
    assert 'def foo():' in result
    assert '# a comment in a function' in result
    assert 'bar()' in result


def test_remove_all_comments():
    result = remove_all_comments(TEST_SOURCE)
    assert 'ignore' not in result
    assert 'nocomment' not in result
    assert 'remove me' not in result
    assert 'import sys' in result


def test_temp_string():
    imports = 'imports\n'
    model = 'model\n'
    data = 'data\n'
    functions = 'functions\n'
    space = 'space'
    result = temp_string(imports, model, data, functions, space)
    assert result == "imports\nfrom hyperopt import fmin, tpe, hp, STATUS_OK, Trials\n" \
                     "functions\ndata\nmodel\n\nspace"


def test_write_temp_files():
    string = 'foo_bar'
    temp_file = './temp.py'
    write_temp_files(string, temp_file)
    assert os.path.isfile(temp_file)
    os.remove(temp_file)


def test_with_line_numbers():
    code = "def do_stuff(x):\n    foo"
    result = with_line_numbers(code)
    print(result)
    assert result == " 1: def do_stuff(x):\n 2:     foo"


def test_determine_indent():
    code = "def do_stuff(x):\n    foo"
    assert determine_indent(code) == '    '
    code = "def do_stuff(x):\n  foo"
    assert determine_indent(code) == '  '
    code = "def do_stuff(x):\n\tfoo"
    assert determine_indent(code) == '\t'


def test_unpack_hyperopt_vals():
    test_vals = {
        'filters_conv_A': [0],
        'filters_conv_B': [1],
        'rate': [0.1553971698387464],
        'units': [1],
        'rate_1': [0.4114807190252343],
        'lr': [2.0215692016654265e-05],
        'momentum': [2],
        'nesterov': [0]
    }
    result = {
        'filters_conv_A': 0,
        'filters_conv_B': 1,
        'rate': 0.1553971698387464,
        'units': 1,
        'rate_1': 0.4114807190252343,
        'lr': 2.0215692016654265e-05,
        'momentum': 2,
        'nesterov': 0
    }
    assert unpack_hyperopt_vals(test_vals) == result


def test_eval_hyperopt_space():
    space = {
        'filters_conv_A': hp.choice('filters_conv_A', [8, 16]),
        'filters_conv_B': hp.choice('filters_conv_B', [16, 24]),
        'rate': hp.uniform('rate', 0, 1),
        'units': hp.choice('units', [96, 128, 192]),
        'rate_1': hp.uniform('rate_1', 0, 1),
        'lr': hp.uniform('lr', 1e-5, 1e-4),
        'momentum': hp.choice('momentum', [0.5, 0.9, 0.999]),
        'nesterov': hp.choice('nesterov', [True, False])
    }
    test_vals = {
        'filters_conv_A': [0],
        'filters_conv_B': [1],
        'rate': [0.1553971698387464],
        'units': [1],
        'rate_1': [0.4114807190252343],
        'lr': [2.0215692016654265e-05],
        'momentum': [2],
        'nesterov': [0]
    }
    test_vals_unpacked = {
        'filters_conv_A': 0,
        'filters_conv_B': 1,
        'rate': 0.1553971698387464,
        'units': 1,
        'rate_1': 0.4114807190252343,
        'lr': 2.0215692016654265e-05,
        'momentum': 2,
        'nesterov': 0
    }
    result = {
        'filters_conv_A': 8,
        'filters_conv_B': 24,
        'rate': 0.1553971698387464,
        'units': 128,
        'rate_1': 0.4114807190252343,
        'lr': 2.0215692016654265e-05,
        'momentum': 0.999,
        'nesterov': True
    }

    assert eval_hyperopt_space(space, test_vals) == result
    assert eval_hyperopt_space(space, test_vals_unpacked) == result


def test_find_signature_end():
    index = find_signature_end(TEST_SOURCE_4)
    assert len(TEST_SOURCE_4) - 10, index
