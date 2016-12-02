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
