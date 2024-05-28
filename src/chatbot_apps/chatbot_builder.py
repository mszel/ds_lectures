"""
A very simple RAG based question answering function.

TODO: question augmentation, graph RAG and many more.
"""

###############################################################################################################
# Import Libraries

# OS related libraries
import sys
from os import path

# temp solution till packaging is done
_homepath = sys.path[0].split('/ds_lectures')[0] + '/ds_lectures'

# own libraries
from src.utils.file_preproc_utils import pprint
from src.utils.llm_utils import get_embedding, llm_call, distances_from_embeddings

# basic ETL libraries
import pandas as pd
import numpy as np


###############################################################################################################
# Helper functions

###############################################################################################################
#  - context generation

def _find_context(
        in_df_knowledge_base: pd.DataFrame, 
        vectorized_question: np.array, 
        max_len: int=1500, _verbose: bool=False) -> str:
    """
    This function finds the context of the question, and returns it.
    """

    _context = ""
    in_df_embeddings = in_df_knowledge_base.copy()
    
    # calculating distance from the question, ordering the table
    in_df_embeddings['q_distance'] = distances_from_embeddings(
        vectorized_question, in_df_embeddings['node_embedding'].values, distance_metric='cosine')
    in_df_embeddings = in_df_embeddings.sort_values(by='q_distance', ascending=True)

    # getting the context (until the limit)
    _curr_len = 0
    _saved_length = 0
    for _idx, _row in in_df_embeddings.iterrows():
        _saved_length = _curr_len
        _curr_len += _row['n_tokens'] + 4
        if (_curr_len < max_len) or _curr_len < 5:
            _context += _row['split_text'] + "\n\n##############################\n\n"
        else:
            break
    
    if _verbose:
        pprint("{} long context is generated.".format(_saved_length))

    return _context

###############################################################################################################
#  - propmpt generation

def _prompt_generation(in_question: str, in_context: str, prompt_list: list, model: str='gpt-3.5-turbo') -> dict:
    """
    This function generates a prompt dictionary, and returns with it.

    TODO: only create and output 1 dictionary.
    """

    # giving a basic return dictionary
    _return_dict = {
        'davinci' : {'prompt':''}, 
        'gpt-3.5-turbo' : {'prompt':[]}, 
        'gpt-4-turbo' : {'prompt':[]}, 
        'curie' : {'prompt':''}
    }
    
    # prompt for non-chatting models
    if model in ['davinci', 'curie']:

        _prompt_text = ""
        
        # adding the system messages
        for _sysmsg in prompt_list:
            if "{context}" in _sysmsg:
                _sysmsg = _sysmsg.format(context=in_context)
            _prompt_text += _sysmsg + "\n\n"
        
        # adding the question to the end
        _prompt_text = _prompt_text + "The question to answer from the context above:{}\n\nMy polite answer: ".format(in_question)
        _return_dict[model].update({'prompt': _prompt_text})
    
    # prompt for chatting models
    elif model in ['gpt-3.5-turbo', 'gpt-4-turbo']:

        _messages = []

        # adding the system messages
        for _sysmsg in prompt_list:
            if "{context}" in _sysmsg:
                _sysmsg = _sysmsg.format(context=in_context)
            _messages.append({'role': 'system', 'content': _sysmsg})
        
        # adding the context and the question
        _messages.append({'role': 'user', 'content': in_question})

        _return_dict[model].update({'prompt': _messages})
    
    return _return_dict


###############################################################################################################
# Main function

def answer_question(in_question: str, cfg: dict, _verbose=False) -> str:
    """
    This function answers a question.
    """
    
    # getting the parameters
    path_in_knowledge_base = path.join(cfg['kb_builder']['path_kb'], 'knowledge_base.pickle')
    max_context_len = cfg['rag_chatbot']['context_size']
    prompt_list = cfg['answer_llm']['prompts']['system_messages']
    model_name = cfg['answer_llm']['modelparams']['model']

    # reading the knowledge base
    _df_knowledge_base = pd.read_pickle(path_in_knowledge_base)

    if _verbose:
        pprint("Knowledge base loaded with {} lines.".format(_df_knowledge_base.shape[0]))

    # vectorizing the question
    _vectorized_question = get_embedding(in_text=in_question, cfg=cfg)

    if _verbose:
        pprint("Question is vectorized. Dimension: {}.".format(len(_vectorized_question)))

    # finding the context
    _context = ""
    if len(_vectorized_question) > 0:
        _context = _find_context(in_df_knowledge_base=_df_knowledge_base, vectorized_question=_vectorized_question, 
                                 max_len=max_context_len, _verbose=_verbose)
    
    # generating the prompt
    _prompt_dict = {}
    if len(_context) > 0:
        _prompt_dict = _prompt_generation(
            in_question=in_question, in_context=_context, prompt_list=prompt_list,
            model=model_name)
    
    # calling the model
    _llm_answer = None
    if len(_context) > 0:
        _llm_answer = llm_call(prompt_dict=_prompt_dict, cfg=cfg, configid='answer_llm')
        
        if _llm_answer:

            # parsing the answer
            if model_name in ['gpt-3.5-turbo', 'gpt-4-turbo']:
                _llm_answer = _llm_answer.choices[0].message.content
            else:
                _llm_answer = _llm_answer.choices[0].text

    # returning the answer
    if _llm_answer:
        _ret_answ = _llm_answer
    else:
        _ret_answ = "Error: please try again: your question was empty or the model had errors."
    
    return _ret_answ