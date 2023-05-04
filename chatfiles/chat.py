import uuid
from file import (check_index_file_exists,
                  get_index_name_from_file_path, get_index_name_from_compress_filepath)
from llm import get_index_by_index_name, create_index, create_graph, get_graph_by_graph_name, get_graph_by_project
from prompt import get_prompt, get_refine_prompt


def check_llama_index_exists(file_name):
    index_name = get_index_name_from_file_path(file_name)
    return check_index_file_exists(index_name)


def create_llama_index(filepath, project):
    index_name = get_index_name_from_file_path(filepath)
    index = create_index(filepath, index_name, project)
    return index_name, index


def create_llama_graph_index(filepaths):
    index_set = {}
    for filepath in filepaths:
        index_name = get_index_name_from_compress_filepath(filepath)
        index = create_index(filepath, index_name)
        index_set[index_name] = index
    graph_name = str(uuid.uuid4())
    graph = create_graph(index_set, graph_name)
    return graph_name, graph


def get_answer_from_index(text, index_name):
    index = get_index_by_index_name(index_name)
    return index.query(text, text_qa_template=get_prompt(), refine_template=get_refine_prompt())


def get_answer_from_graph(text, graph_name):
    graph = get_graph_by_graph_name(graph_name)
    return graph.query(text)


def get_answer_from_project(text, project):
    graph = get_graph_by_project(project)
    # custom_query_engines = {}
    # for id, index in graph.all_indices.items():
    #     query_engine = index.as_query_engine(text_qa_template=get_prompt(), refine_template=get_refine_prompt())
    #     custom_query_engines[id] = query_engine
    # query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "text_qa_template": get_prompt(),
                "refine_template": get_refine_prompt()
            }
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "text_qa_template": get_prompt(),
                "refine_template": get_refine_prompt()
            }
        }
    ]
    return graph.query(text, query_configs=query_configs)
