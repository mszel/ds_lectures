"""
Functions for preprocessing pdf files.

TODO: Add it to a class that has standardized functions (as all docu loaders).
Or just use langchain, or similar.
"""

###############################################################################################################
# Import Libraries                                                                                      #

# OS related libraries
import sys

# pdf cleaning
from PyPDF2 import PdfReader

# temp solution till packaging is done
_homepath = sys.path[0].split('/ds_lectures')[0] + '/ds_lectures'

# own libraries
from src.utils.file_preproc_utils.common_prep_utils import _rawtext_cleaner


###############################################################################################################
# PDF handler functions (TODO: add a more complex one) 

def handle_pdf(path_in_pdf: str, spec_chars: list=[]) -> str:
    """
    Handle pdf files.
    """
    _raw_text = ''

    pdffileobj=open(path_in_pdf, 'rb')
    _allpages = PdfReader(pdffileobj)
    _totpages = _allpages.numPages
    for _page in range(_totpages):
        pagehandle = _allpages.getPage(_page)
        _raw_text += pagehandle.extractText() + '\n\n'
    
    # do the basic cleaning
    _raw_text = _rawtext_cleaner(_raw_text, spec_chars)
    return _raw_text