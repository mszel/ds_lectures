"""
Functions for generating embeddings.

TODO: add parallelization (async).
TODO: generate a class from this.
"""

###############################################################################################################
# Import Libraries

# OS related libraries
import sys
from time import sleep
from typing import List

# openAI API related functions
from openai import OpenAI

# basic ETL libraries
import numpy as np

# distance calculation
from scipy import spatial

###############################################################################################################
# Embedders 

def get_embedding(in_text: str, cfg: dict) -> np.array:
    """
    Get the embedding of a text using openAI's models.

    TODO: add more embedder providers.
    TODO: input can be a list of texts, output can be a list of embeddings.
    """

    # get a client (TODO: the class will handle it in the init)
    _embedder_cli = None
    if cfg['embedder']['provider'] == 'openai':
        _login_params = cfg['embedder']['login']
        _embedder_cli = OpenAI(**_login_params)


    _return_val = np.array([])
    if len(in_text.replace("\n", " ").strip()) > 0:
        if cfg['embedder']['provider'] == 'openai':
            text_m = in_text.replace("\n", " ")
            _curr_try = 0
            _solved = False

            # trying to get an answer
            while ((not _solved) and (_curr_try < cfg['embedder']['maxretry'])):
                try:
                    sleep(cfg['embedder']['sleeptime']) # to avoid openAI error (429-alike)
                    _return_val = _embedder_cli.embeddings.create(input=text_m, model=cfg['embedder']['modelparams']['model'])
                    _return_val = _return_val.data[0].embedding
                    _solved = True
                except Exception as e:
                    print('Error while calling openAI API... Trying in 1 sec...')
                    print(e)
                    _curr_try += 1
        else:
            raise ValueError('Model not supported.')

    return _return_val


###############################################################################################################
# Further util functions related to embeddings 

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances
