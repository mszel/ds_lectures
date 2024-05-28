"""
Functions for chat completion.

TODO: add parallelization (async).
TODO: generate a class from this.
"""

###############################################################################################################
# Import Libraries

# OS related libraries
import sys
from time import sleep

# openAI API related functions
from openai import OpenAI

# basic ETL libraries
import numpy as np


###############################################################################################################
# Chat completion

def llm_call(prompt_dict: dict, cfg: dict, configid='answer_llm') -> dict:
    """
    Calling a large language model, described in the model variable. The prompt dict
    should contain a prompt with the key of the selected model.
    """

    # get a client
    _ccpl_cli = None
    if cfg[configid]['provider'] == 'openai':
        _login_params = cfg[configid]['login']
        _ccpl_cli = OpenAI(**_login_params)
    
    # getting the parameters
    model_name = cfg[configid]['modelparams']['model']
    version_num = cfg[configid]['version_num']
    maxretry_num = cfg[configid]['maxretry_num']

    # generating base parameter dictionary
    _prompt = prompt_dict[model_name]['prompt']

    if model_name in ['davinci', 'curie']:
        _model_params = {'prompt': _prompt} # type: ignore - covered in all cases
    elif model_name in ['gpt-3.5-turbo', 'gpt-4-turbo']:
        _model_params = {'messages': _prompt} # type: ignore - covered in all cases
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    # adding extra parameters
    _model_params.update(cfg[configid]['modelparams'])
    if version_num > 1:
        _model_params.update({'n': version_num})
    
    # CALLING THE LLM
    _llm_answer = None

    # openai ChatCompletion cases
    if model_name in ['gpt-3.5-turbo', 'gpt-4-turbo']:
        
        # creating an empty answer, init mid-variables
        _curr_try = 0
        _solved = False

        # trying to get an answer
        while ((not _solved) and (_curr_try < maxretry_num)):
            try:
                _llm_answer = _ccpl_cli.chat.completions.create(**_model_params)
                _solved = True
            except Exception as e:
                print(f"The following error catched while calling Open AI API: {e}. Retrying in 1 second...")
                sleep(1)
                _curr_try += 1
    
    elif model_name in ['davinci', 'curie']:

        # creating an empty answer, init mid-variables
        _curr_try = 0
        _solved = False

        # trying to get an answer
        while ((not _solved) and (_curr_try < maxretry_num)):
            try:
                _llm_answer = _ccpl_cli.completions.create(**_model_params)
                _solved = True
            except Exception as e:
                print(f"The following error catched while calling Open AI API: {e}. Retrying in 1 second...")
                sleep(1)
                _curr_try += 1
    
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    # returning the answer
    return _llm_answer
