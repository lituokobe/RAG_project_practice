from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from RAG_project_practice.documents.markdown_parser import MarkdownParser
from RAG_project_practice.documents.milvus_db import MilvusVectorSave
from RAG_project_practice.llm_models.embedding_model import bge_embedding
from RAG_project_practice.utils.env_utils import MILVUS_URI, COLLECTION_NAME


def test1():
    #1. search with MilvusVectorSave using ANN method to search dense vectors
    mv = MilvusVectorSave()
    mv.create_connection(is_first = False)

    # result = mv.vector_store_saved.similarity_search(
    #     #this function only returns the search result: page_content and metadata
    #     query = '现在最先进的纳米级清洗技术是什么？',
    #     k = 2,#maximum results to return
    #     #expr='category =="Title"', #optional, if you only want titles to be returned
    #     # expr = 'category =="content"'  # optional, if you only want content to be returned
    # )

    result = mv.vector_store_saved.similarity_search_with_score(
        #this function can return results, but with similarity score
        #Tt returns a tuple, the first one is Document (including metadata, page_content, title),
        # the second one is the score
        query = '现在最先进的纳米级清洗技术是什么？',
        k = 2,#maximum results to return
        #expr='category =="Title"', #optional, if you only want titles to be returned
        expr = 'category =="content"'  # optional, if you only want content to be returned
    )

    for doc in result:
        print(doc)


def test2():
    """
    create a new collection for full-text search with sparse vectors ONLY
    :return:
    """
    client = MilvusClient(uri = MILVUS_URI)

    schema = client.create_schema()
    schema.add_field(field_name='id', datatype = DataType.INT64, is_primary = True, auto_id = True)
    schema.add_field(field_name='text',
                     datatype=DataType.VARCHAR,
                     max_length=6000,
                     # enable_analyzer is activated and we need to set it to Chinese,
                     # the chinese analyzer is different from the english one.
                     # if you don't set it to Chinese, the search function won't work
                     enable_analyzer=True,
                     # analyzer_params = {'type':'chinese'},
                     analyzer_params = {'tokenizer': 'jieba', 'filter': ['cnalphanumonly']} #jieba is a better Chinese analyzer
                     )

    # we won't do full-text search for category, so no need to enable analyzer, otherwise it will be slow.
    schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)

    #Add a function using BM25 for to convert text to sparse vectors
    bm25_function = Function(
        name="text_bm25_emb",  # Function name, any value
        input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse"],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,  # Set to `BM25`
    )
    schema.add_function(bm25_function)
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )

    if 'demo' in client.list_collections():
        # release, delete index, then delete collection
        client.release_collection(collection_name='demo')
        client.drop_index(collection_name='demo', index_name='sparse_inverted_index')
        client.drop_collection(collection_name='demo')

    client.create_collection(
        collection_name='demo',
        schema=schema,
        index_params=index_params
    )

def test3():
    """insert data to the collection with sparse vectors ONLY"""
    vector_store = Milvus( #this is from langchain-milvus library, this one doesn't support setting schema
        embedding_function=None, #this one doesn't have dense vectors, no need to set up this one
        collection_name='demo',
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field=['sparse'],
        consistency_level="Strong",
        auto_id=True,
        connection_args={"uri": MILVUS_URI}
    )

    #parse the documents:
    file_path = '../datas/md/tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    #add the parsed documents to vector_store:
    vector_store.add_documents(docs)
    #though in the parsed documents, there are more than 4 keys (of the collections)
    #so long as the 4 keys can be found in the parsed docs, they will be added.
    #the rest keys in the documents will be ignored

def test4():
    """full text search"""
    vector_store = Milvus(
        embedding_function=None,
        collection_name='demo',
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field=['sparse'],
        consistency_level="Strong",
        auto_id=True,
        connection_args={"uri": MILVUS_URI}
    )
    #full text search has the same code as ANN, except it's searching the sparse vectors
    result = vector_store.similarity_search_with_score(
        query = '活性氧原子',
        k=2,
    )

    for doc in result:
        print(doc)


def test5():
    """Use PyMilvus library to query the collection"""
    client = MilvusClient(uri=MILVUS_URI)
    res = client.search(
        collection_name='demo',
        data = ['半导体表面特征改善'],
        anns_field='sparse',
        limit=3,
        output_fields=['text', 'category', 'id'],
        search_params={'params': {'drop_ratio_search':0.2}} #the percentage of ignoring low importance
    )
    for doc in res[0]:
        print(doc)

def test7():
    """hybrid search with pymilvus"""
    search_params_1 = { #parameters for dense vector search
        'data': [bge_embedding.embed_query('现在最先进的纳米级清洗技术是什么？')], #embed the question you want to search
        'anns_field': 'dense',
        'param': {
            'metric_type': 'IP',
            'params': {'nprobe': 10},
        },
        'limit':5
    }
    req1 = AnnSearchRequest(**search_params_1) # sparse vector search

    search_params_2 = {  # parameters for sparse vector search
        'data': ['现在最先进的纳米级清洗技术是什么？'], #directly pass the query
        'anns_field': 'sparse',
        'param': {
            'metric_type': 'BM25',
        },
        'limit': 5,

    }
    req2 = AnnSearchRequest(**search_params_2)  # sparse vector search

    client = MilvusClient(uri=MILVUS_URI)
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2], #list of search request object
        ranker=RRFRanker(60), #k=60 for RRF, means the result is less various
        limit=5,
        output_fields=['text', 'category', 'title'],
    )
    for hits in res:
        print(f'Top N result:')
        for item in hits:
            print(item)

def test8():
    """hybrid search with langchain-milvus"""
    mv = MilvusVectorSave()
    mv.create_connection()
    res = mv.vector_store_saved.similarity_search_with_score(
        #.vector_store_saved is a langchain Milvus object, it can do search by default
        query = '现在最先进的纳米级清洗技术是什么？',
        k=3,
        ranker_type='rrf',
        # ranker_type='weighted',
        ranker_params={'k':100}, ##k=100 for RRF, means the result is more various
    )

    for item in res:
        print(item)

def test9():
    """hybrid search with langchain-milvus"""
    mv = MilvusVectorSave()
    mv.create_connection()
    retriever = mv.vector_store_saved.as_retriever(
        # search type possible values:
        # - 'similarity' (default)
        # - 'mmr'
        # - 'similarity_score_threshold' (not support hybrid search)
        search_type='similarity', #only return results that passed the threshold
        search_kwargs = {
            'k':3,
            "score_threshold": 0.1,
            "ranker_type":"rrf",
            "ranker_params":{'k':100},
            'filter':{'category': 'content'} #add a filter, just a simple show-off.
        }
    )

    res = retriever.invoke('光刻机有哪几种？')

    for item in res:
        print(item)


if __name__ == '__main__' :
    # test2() #create the collection structure

    # test3() #insert data

    # test4() #full-text search with Milvus from langchain-milvus

    # test5() #fullt-text search with MilvusClient from pymilvus

    # test7() #hybryd search with MilvusClient from pymilvus

    # test8() #hybryd search with Milvus from langchain-milvus

    test9() #hybryd search with Milvus from langchain-milvus, but as langchain runnable, ready for future chain building

