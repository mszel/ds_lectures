"""
Loads the application's configuration from yaml file into a dictionary.
Handles environment variable placeholders in the configuration file, as
well as prompt name refereces from a central prompt JSON file.
"""

import yaml
import re
import os
import json
import sys

# temp solution till packaging is done
_homepath = sys.path[0].split('/ds_lectures')[0] + '/ds_lectures'

# place of promptfile (later can be configurable)
path_chatbot_propmts = _homepath + '/src/llm_prompts/chatbotprompts.JSON'
with open(path_chatbot_propmts, 'r') as json_file:
        CHB_PROMPT_DICT = json.load(json_file)


# Regular expression to match environment variable placeholders
path_matcher = re.compile(r'\$\{([^}^{]+)\}')

def path_constructor(loader, node):
    """ Custom constructor for handling environment variable placeholders. """
    value = loader.construct_scalar(node)
    env_var = path_matcher.match(value).group(1)
    return os.getenv(env_var, 'not found')

def prompt_constructor(loader, node):
    # Load JSON file containing the prompts
    key = loader.construct_scalar(node)
    # Return the prompt text corresponding to the key
    return CHB_PROMPT_DICT.get(key, [])

def datapath_constructor(loader, node):
    # Load JSON file containing the prompts
    rel_path = loader.construct_scalar(node)
    # Return the prompt text corresponding to the key
    return _homepath + '/' + rel_path

# Add resolver and constructor to the YAML loader
yaml.add_implicit_resolver('!path', path_matcher, None, yaml.SafeLoader)
yaml.add_constructor('!path', path_constructor, yaml.SafeLoader)

# Add resolver and constructor to the YAML loader
yaml.add_constructor('!chatprompt', prompt_constructor, yaml.SafeLoader)
yaml.add_constructor('!datapath', datapath_constructor, yaml.SafeLoader)

def load_config(config_file: str) -> dict:
    with open(config_file, "r") as file:
        return yaml.safe_load(file)