import os
import openai
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from llama_index import ComposableGraph, GPTListIndex, LLMPredictor, GPTSimpleVectorIndex, ServiceContext, \
    SimpleDirectoryReader, download_loader

from file import check_index_file_exists, get_index_filepath, get_name_with_json_extension
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding

from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

llm_predictor = LLMPredictor(llm=AzureChatOpenAI(
    deployment_name="gpt-35-turbo"))

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(),
    embed_batch_size=1,
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm)


def create_remote_index(url, project):
    RemoteDepthReader = download_loader("RemoteDepthReader")

    loader = RemoteDepthReader(domain_lock=True)
    documents = loader.load_data(url)
    print(f"Found {len(documents)} documents.")
    index = GPTSimpleVectorIndex.from_documents(documents[:5], service_context=service_context)

    data, count = supabase.table('files').insert(
        { "project_id": project, "path": url, "meta": index.save_to_dict() }
        ).execute()
    return index


def create_index(filepath, index_name, project):
    index = get_index_by_index_name(index_name)
    if index is not None:
        return index

    # index_name = get_name_with_json_extension(index_name)
    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    data, count = supabase.table('files').insert(
        { "project_id": project, "path": index_name, "meta": index.save_to_dict() }
        ).execute()
    # index.save_to_disk(get_index_filepath(index_name))
    return index


def get_index_by_index_name(index_name):
    index_name = get_name_with_json_extension(index_name)
    if check_index_file_exists(index_name) is False:
        return None
    index_filepath = get_index_filepath(index_name)
    index = GPTSimpleVectorIndex.load_from_disk(index_filepath, service_context=service_context)
    return index


def create_graph(index_sets, graph_name):
    graph_name = get_name_with_json_extension(graph_name)
    graph = ComposableGraph.from_indices(GPTListIndex,
                                         [index for _, index in index_sets.items()],
                                         index_summaries=[f"This index contains {indexName}" for indexName, _ in index_sets.items()],
                                         service_context=service_context)
    graph.save_to_disk(get_index_filepath(graph_name))
    return graph


def get_graph_by_graph_name(graph_name):
    graph_name = get_name_with_json_extension(graph_name)
    graph_path = get_index_filepath(graph_name)
    graph = ComposableGraph.load_from_disk(graph_path, service_context=service_context)
    return graph


def get_graph_by_project(project):
    data = supabase.table("files").select("*").eq("project_id", project).execute()
    index_sets = {}
    for file in data.data:
        index = GPTSimpleVectorIndex.load_from_dict(file['meta'])
        index_sets[file['path']] = index
    graph = ComposableGraph.from_indices(GPTListIndex,
                                         [index for _, index in index_sets.items()],
                                         index_summaries=[f"This index contains {indexName}" for indexName, _ in index_sets.items()],
                                         service_context=service_context)
    return graph
