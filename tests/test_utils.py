import os
from hyperas.utils import (
    extract_imports, remove_imports, remove_all_comments, temp_string,
    write_temp_files, with_line_numbers, determine_indent)

TEST_SOURCE = """
from __future__ import print_function
from sys import path
import os
import sys # ignore this comment
''' remove me '''
# import nocomment
from java.lang import stuff
from _pydev_ import stuff
"""

TEST_SOURCE_2 = """
import sys
foo_bar()
"""


def test_extract_imports():
    result = extract_imports(TEST_SOURCE)
    assert 'java.lang' not in result
    assert 'nocomment' not in result
    assert '_pydev_' not in result
    assert 'try:\n    import os\nexcept:\n    pass\n' in result
    assert 'from sys import path' in result
    assert 'ignore' not in result
    assert 'remove me' not in result
    assert 'from __future__ import print_function' in result


def test_remove_imports():
    result = remove_imports(TEST_SOURCE_2)
    assert 'foo_bar()' in result


def test_remove_all_comments():
    result = remove_all_comments(TEST_SOURCE)
    assert 'ignore' not in result
    assert 'nocomment' not in result
    assert 'remove me' not in result
    assert 'import sys' in result


def test_temp_string():
    imports = 'imports\n'
    data = 'data\n'
    model = 'model\n'
    space = 'space'
    result = temp_string(imports, model, data, space)
    assert result == "imports\nfrom hyperopt import fmin, tpe, hp, STATUS_OK, Trials\nfrom hyperas.distributions import conditional\ndata\nmodel\n\nspace"


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
