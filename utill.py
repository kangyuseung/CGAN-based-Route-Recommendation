import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.decomposition import PCA, KernelPCA



def User_embedding_Func(g, model):
    node_ids=model.wv.index_to_key
    x=(model.wv.vectors)
    y = [g.node_type(node_id) for node_id in node_ids]

    node_embedding = pd.DataFrame(x, index = node_ids)
    node_embedding['target'] = y

    #User Embedding
    User_embedding = node_embedding[node_embedding['target']=='user']

    del User_embedding['target']
    User_embedding.index.name = 'USER_ID'

    return User_embedding

def Route_embedding_Func(Route_sequence):
    Route_sequence.columns=["Route_Name", "Place"]
    Route_sequence_nan = Route_sequence[Route_sequence['Place'].str.contains("nan", na = True, case=False)]
    Route_sequence = Route_sequence[Route_sequence['Place'].isin(Route_sequence_nan['Place'])== False]

    places = (Route_sequence['Place'])

    word_to_index = {}
    index_to_word = {}
    current_index = 0

    sequences = []
    for place in places:
        sequence = []
        for word in place.split(", "):
            if word not in word_to_index:
                word_to_index[word] = current_index
                index_to_word[current_index] = word
                current_index += 1
            sequence.append(word_to_index[word])
        sequences.append(sequence)

    max_sequence_length = max(len(sequence) for sequence in sequences)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

    embedding_dim = 32  
    embedding_output_dim = 64  

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(word_to_index), output_dim=embedding_dim, input_length=max_sequence_length),
        tf.keras.layers.LSTM(units=embedding_dim, return_sequences=False),  
        tf.keras.layers.Dense(embedding_output_dim)  
    ])

    RNN_embedded_data = model.predict(padded_sequences)

    Route_embedding = pd.DataFrame(RNN_embedded_data, index=Route_sequence['Route_Name'])
    Route_embedding.index.name = 'ROUTE_ID'

    return Route_embedding