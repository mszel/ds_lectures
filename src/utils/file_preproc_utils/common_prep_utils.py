"""
Common functions used by all preprocessors.

TODO: add here the basic of the file preprocessor class as well.
"""

###############################################################################################################
# Import Libraries

import time
import re

###############################################################################################################
# util functions

def pprint(s: str) -> None:
    """ 
    Pretty print: writing out the time while print the input string.
    """
    print('[' + time.strftime('%a %H:%M:%S') + '] ' + s)
    pass

###############################################################################################################
# document preprocessing functions

def _string_cleaner(in_str: str, spec_chars: list=[]) -> str:
    """
    Cleaning the string from non-unicode and non alphanumeric characters.
    Keeping the listed characters from the spec_chars input list.
    """

    _specdef = ['@', '#', '$', '%', '&', '*', '(', ')', '/', '"', '\'', 
                '「', '」', '|', '-', ':', ' ', ',', '.', '!', '?', '[', 
                ']', '{', '}', '<', '>', '=', '+', '~', '`', '^', ';', 
                '\n', '。', '、', ',', '\t'] + spec_chars

    _goodchars = re.findall("[^\W]", in_str, re.UNICODE)
    _goodchars = list(set(_specdef + _goodchars))

    return ''.join([_i for _i in in_str if _i in _goodchars])


def _rawtext_cleaner(in_str: str, spec_chars: list=[]) -> str:
    """
    Removing extra new lines, tabs, white spaces, etc - as well as the
    non-unicode characters.
    """
    
    _return_txt = ""
    _raw_text = re.sub(r'\n{2,}', '\n\n', in_str)
    _raw_text = re.sub(r'[ \t]+', ' ', _raw_text)
    _raw_text = _raw_text.split('\n\n')
    _raw_text = [_string_cleaner(_i, spec_chars) for _i in _raw_text]
    _raw_text = [_i for _i in _raw_text if len(_i.strip()) >= 2]
    
    if len(_raw_text) > 0:
        _return_txt = '\n\n'.join(_raw_text)
    
    return _return_txt
