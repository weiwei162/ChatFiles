from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

DEFAULT_PROMPT = (
    "We have provided context information below: \n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Please answer the question in Chinese based on the above information."
)


def get_prompt():
    return QuestionAnswerPrompt(DEFAULT_PROMPT)

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question in Chinese."
)


def get_refine_prompt():
    return RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)
