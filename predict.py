import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.decomposition import PCA, KernelPCA

from init import HIN_graph, HIN_embedding
from utill import User_embedding_Func, Route_embedding_Func

#Data Input
route_review = pd.read_csv('Data/tripadvisor_route_review.csv')
route = pd.read_csv('Data/tripadvisor_route.csv')
data=route.merge(route_review, on='ROUTE_ID')

#HIN Modeling
g = HIN_graph(data)
model = HIN_embedding(g)

User_embedding = User_embedding_Func(g, model)


#Route Embedding
Route_sequence = pd.read_csv('Data/tripadvisor_route_list.csv')
Route_embedding = Route_embedding_Func(Route_sequence)


#Evaluation 

temp = route_review.merge(User_embedding, on = 'USER_ID')
temp = temp.merge(Route_embedding, on = 'ROUTE_ID')
#temp = temp.drop_duplicates()

Feature_vec = temp[list(temp.columns[3:])].to_numpy()
label = temp['Rating'].to_numpy()

pca = PCA(n_components=128, random_state = 150)

Latent_vector = pca.fit_transform(Feature_vec)


from sklearn.model_selection import train_test_split
training_data, test_data , training_labels, test_labels = train_test_split(Latent_vector, label, test_size = 0.2, shuffle = label, random_state = 150)



import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, multiply, Dense, Flatten, Concatenate
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

X = training_data
y = training_labels  # Label column

# Parameters
num_features = X.shape[1]
num_labels = len(np.unique(y))+1
latent_dim = 128

# Define the generator
def build_generator():
    model = Sequential()

    model.add(Dense(128, input_dim=latent_dim))
    model.add(Dense(256))
    model.add(Dense(num_features, activation='linear'))

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_labels, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    output = model(model_input)

    return Model([noise, label], output)

# Define the discriminator
def build_discriminator():
    img = Input(shape=(num_features,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_labels, num_features)(label))

    model_input = Concatenate(axis=1)([img, label_embedding])

    model = Sequential()

    model.add(Dense(128, input_dim=num_features + num_features))
    model.add(Dense(64))
    model.add(Dense(1, activation='sigmoid'))

    validity = model(model_input)

    return Model([img, label], validity)

# Build and compile the generator
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5))

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5), metrics=['accuracy'])

# Build the combined model
z = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([z, label])
discriminator.trainable = False
valid = discriminator([img, label])

combined = Model([z, label], valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5))

# Train the model
def train(epochs, batch_size=128):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0], batch_size)
        imgs, labels = X[idx], y[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        #print('noise')
        #print(noise)
        gen_imgs = generator.predict([noise, labels.reshape(-1, 1)])

        d_loss_real = discriminator.train_on_batch([imgs, labels.reshape(-1, 1)], real_labels)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels.reshape(-1, 1)], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        sampled_labels = np.random.randint(0, num_labels, batch_size).reshape(-1, 1)
        g_loss = combined.train_on_batch([noise, sampled_labels], real_labels)

        print(f"{epoch+1} [D loss: {d_loss[0]:.2f}, accuracy: {100 * d_loss[1]:.2f}] [G loss: {g_loss:.2f}]")

train(epochs=500)

def generate_samples(num_samples, labels):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    gen_data = generator.predict([noise, labels.reshape(-1, 1)])
    return gen_data


num_samples_to_generate=15000
generated_label = np.array([(i % 5) + 1 for i in range(num_samples_to_generate)])

generated_data = generate_samples(num_samples=len(generated_label), labels=generated_label)


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# MLP Regressor 
mlp = MLPRegressor(hidden_layer_sizes=(150,100,50), max_iter=50, alpha=1e-4, solver='sgd', verbose=0, random_state=150, learning_rate_init=0.01)
mlp.fit(generated_data, generated_label)
mlp_pred = mlp.predict(test_data)


rmse = mean_squared_error(mlp_pred, test_labels)**0.5
mae = mean_absolute_error(mlp_pred, test_labels)

print(rmse)
print(mae)