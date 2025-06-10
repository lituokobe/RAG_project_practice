from pymilvus import MilvusClient
import numpy as np

client = MilvusClient(uri = 'http://127.0.0.1:19530/') #port 19530 is for database operation

#create a collection
client.create_connection(
    collection_name='demo_collection',
    dimension=384, # the dimension of the vector. each vector is an array of 384 floats
)

docs = [ #define some documents to be converted
    'Artificial intelligence was founded as academic discipline in 1956',
    'Alan Turing was the first person to conduct substantial research in AI',
    'Born in Maida Vale, London, Turing was raised in southern England.'
]

vectors = [[np.random.uniform(-1, 1) for _ in range (384)] for _ in range(len(docs))] # only generate vectors from random selecting numbers

# format the data in to dictionaries and then in a list
data = [
    {'id': i, 'vector': vectors[i], 'text': docs[i], 'subject': 'AI history'} for i in range(len(docs))
]

#insert the data to the created collections
res = client.insert(collection_name='demo_collection', data = data)

print('Insert result', res)

#release the collection, there won't be anything in the RAM now. The following operation cannot be done until load_collection.
client.release_collection(collection_name='demo_collection')

# #search the vectors based on similarity
# res = client.search(
#     collection_name='demo_collection',
#     data=[vectors[0]], #search the vectors, we use the first vector (not possible to use a new one, as the vectors are generated randomly)
#     filter = 'subject == "AI history"',
#     limit = 2,
#     output_fields= ['text', 'subject']
# )
#
# print('Search result', res)

#load the collection
client.load_collection(collection_name='demo_collection')

#query the vectors with filter
res = client.query(
    collection_name='demo_collection',
    filter = 'subject == "AI history"',
    output_fields= ['text', 'subject']
)

print('Query result', res)