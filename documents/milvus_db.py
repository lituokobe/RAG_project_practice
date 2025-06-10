from typing import List
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from RAG_project_practice.documents.markdown_parser import MarkdownParser
from RAG_project_practice.llm_models.embedding_model import bge_embedding
from RAG_project_practice.utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """
    save new document data to the database
    """
    def __init__(self):
        """
        Define collection's index
        """
        self.vector_store_saved: Milvus = None
        # self.index_params = [
        #     { #dense vector
        #         "field_name" :"dense",
        #         "index_name" : "dense_vector_index",
        #         #HNSW organizes vector data in a multi-layered graph structure, making search operations faster
        #         "index_type" : IndexType.HNSW,
        #         "metric_type" : MetricType.IP, #IP: inner product, L1: Euro distance
        #         "params" : {"M":16, "efConstruction":64}
        #         #M: node numbers to connect. the bigger, the more accurate, but more space of RAM needed
        #         #efConstruction: search scope: between 50-200
        #     },
        #     { #sparse vector
        #         "field_name" : "sparse",
        #         "index_name" : "sparse_inverted_index",
        #         "index_type" : "SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        #         "metric_type" : "BM25",
        #         "params" : {
        #             "inverted_index_algo": "DAAT_MAXSCORE",
        #             "bm25_k1": 1.6,
        #             "bm25_b": 0.75
        #         },
        #     },
        # ]

    def create_collection(self):
        """
        create a new collection for full-text search with sparse vectors ONLY
        :return:
        """
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text',
                         datatype=DataType.VARCHAR,
                         max_length=6000,
                         enable_analyzer=True,
                         analyzer_params={'tokenizer': 'jieba', 'filter': ['cnalphanumonly']}
                         )

        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        # Add a function using BM25 for to convert text to sparse vectors
        bm25_function = Function(
            name="text_bm25_emb",  # Function name, any value
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index( #index for sparse
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

        index_params.add_index( #index for dense
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M: count of near node to connet, efConstruction: search scope
        )

        if COLLECTION_NAME in client.list_collections():
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """
        create a connection: milvus + langchain
        :return:
        """
        # if is_first: #only applicable when create the collection for the first time
        #     client = MilvusClient(uri=MILVUS_URI)  # The MilvusClient is from pymilvus, the original Milvus library
        #     # Check if there is already a collection
        #     if COLLECTION_NAME in client.list_collections():
        #         # release the collection (from the RAM) before deleting it
        #         client.release_collection(collection_name=COLLECTION_NAME)
        #         #if you don't release the collection from the RAM, there will be error to drop_index
        #         client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_vector_index')
        #         client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
        #         client.drop_collection(collection_name=COLLECTION_NAME)

        self.vector_store_saved = Milvus(#This milvus instance is from langchain_milvus
            #Milvus is an unstructured database, meaning the fields of the data are not necessarily consistent
            #Here, we only have 2 keys added to the vector: dense, sparse
            #other keys will be added on the go if there are other keys in the data or metadata
            embedding_function=bge_embedding, #dense vector
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(), #sparse vector
            vector_field=['dense', 'sparse'],
            # index_params=self.index_params,
            consistency_level="Strong", #highest level of consistency: Strong > Session > Bounded > Eventually
            auto_id=True,
            connection_args={"uri": MILVUS_URI},
        )

    def add_documents(self, datas:List[Document]):
        """"
        add new documents to the collection
        """
        self.vector_store_saved.add_documents(datas)


if __name__ == '__main__': #test the class created
    file_path = '../datas/md/tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    #create the collection
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    #get the structure of the collection
    client = mv.vector_store_saved.client
    desc_collection = client.describe_collection(
        collection_name = COLLECTION_NAME
    )
    #check the collection
    print('The collection structure is:', desc_collection)

    # get all the index from the collection
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('All index in the collection', res)

    if res:
        for i in res:
            # get index description
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    #query from teh collection
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # check all the data where category == 'Title'
        # in the output fields, 'id' is returned by default, others need to be specified.
        output_fields=['text', 'category', 'filename', ]  # specify returned field
    )

    print('Testing, the query result with filter is: ', result)
