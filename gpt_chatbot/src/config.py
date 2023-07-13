# chatbot prompts

PROMPT_BASE_DICT = {
    'book_robot': {
        'system_messages': [
            "You are answering questions related to articles, webpages and books that are loaded to your knowledge base.",
            "Answer the question carefully based on the provided context below. If there is no answer in the provided context, you can still answer the question, but warn the user that the answer is not in the provided context.", 
            "Answer the questions on the same language as the question is asked.",
         ]} 
}


# openai parameters
MODEL_PARAMS = {
    'davinci': {
        'model':'davinci',
        'temperature': 0.8,
        'max_tokens': 512,
        'top_p': 1,
        'frequency_penalty': 0.0
    },
    'curie': {
        'model':'curie',
        'temperature': 0.8,
        'max_tokens': 512,
        'top_p': 1,
        'frequency_penalty': 0.0
    },
    'gpt3.5': {
        'model':'gpt-3.5-turbo',
        'temperature': 0.8,
        'max_tokens': 512,
        'top_p': 1,
        'frequency_penalty': 0.0
    },
    'gpt4': {
        'model':'gpt-4',
        'temperature': 0.8,
        'max_tokens': 512,
        'top_p': 1,
        'frequency_penalty': 0.0
    },
}


