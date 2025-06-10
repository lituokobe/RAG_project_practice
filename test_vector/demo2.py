from pymilvus import MilvusClient, Function, FunctionType, DataType

client = MilvusClient(uri = 'http://127.0.0.1:19530/') #active Milvus database on Docker, port 19530 is for database operation+

"""
In Milvus, create_schema() is used to define the structure of your collection—essentially,
you're specifying how data should be stored and queried.
"""
schema = client.create_schema() #create
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR) #store value after sparse embedding

# function for sparse embedding: read original data, convert it to vector with BM25, save the sparse vector to the output
bm25_function = Function(
    name="text_bm25_emb", # Function name
    input_field_names=["text"], # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    function_type=FunctionType.BM25, # Set to `BM25`
)

schema.add_function(bm25_function)

# configure index
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX", # Inverted index type for sparse vectors
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE", # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
        "bm25_k1": 1.6,  # range: [1.2 ~ 2.0], the bigger the value, the higher ranking for professional terminologies
        "bm25_b": 0.75 #range[0,1], 0: fully normalized, 1: not normalized at all
    },
)

# create a collection
client.create_connection(
    collection_name='t_demo2',
    schema=schema,
    index_params=index_params
)

# insert test data
client.insert('t_demo2', [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
])

# Starting match search (full-text search)
search_params = {
    'params': {'drop_ratio_search': 0.2},  # % to ignore low-importance words: least important (similarity) 20% words in the query vector will be ignored during searching
}

resp = client.search(
    collection_name='t_demo2',
    data=['whats the focus of information retrieval?'],
    anns_field='sparse',  # sparse vectors to match
    limit=3,
    search_params=search_params,
    output_fields=["text"]
)

print(resp)


