"""
Functions for preprocessing text files.

TODO: Add it to a class that has standardized functions (as all docu loaders).
Or just use langchain, or similar.
"""

###############################################################################################################
# Import Libraries                                                                                    #

# OS related libraries
import sys

# temp solution till packaging is done
_homepath = sys.path[0].split('/ds_lectures')[0] + '/ds_lectures'

# own libraries
from src.utils.file_preproc_utils.common_prep_utils import _rawtext_cleaner


###############################################################################################################
# Text handler functions (TODO: add a more complex one) 

def handle_text(path_in_txt: str, _encoding: str, spec_chars: list=[]) -> str:
    """
    Handling text files.
    """

    with open(path_in_txt, encoding=_encoding, errors='ignore') as f:
        _raw_text = '\n'.join(f.readlines())
    
    # do the basic cleaning
    _raw_text = _rawtext_cleaner(_raw_text, spec_chars)
    return _raw_text