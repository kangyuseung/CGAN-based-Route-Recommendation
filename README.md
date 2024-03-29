# Conditional Generative Adversarial Network based Travel Route Recommendation

In this study, we propose a route recommendation model using a Conditional Generative Adversarial Network (CGAN), based on the extraction of user characteristics and route features. Tourist routes have unique attributes distinct from traditional products like movies or books. To overcome this, we extract features from the relationship between users and routes, including locations, and from the route characteristics represented by the sequence of places, thereby constructing a latent vector. To address the data sparsity issue in the relationship between routes and users, we train the CGAN using the latent vector and user preference data for routes. The generated samples are then used to predict user route preferences. To evaluate the effectiveness of our proposed model, we employ two real-world tourist route datasets. Experimental results show that our model records up to approximately 30.53% improvement in Mean Absolute Error (MAE) and 26.91% in Root Mean Squared Error (RMSE) compared to the best-performing models. Additionally, in further experiments conducted in sparse data environments, our model demonstrates adaptability with about 30.75% and 18.43% improvements in MAE and RMSE, respectively, compared to the best-performing models, thereby confirming its efficacy in sparse data environments.

## Requirements

- Python 3.7.9
- stellargraph > 1.2.1
- tensorflow > 2.11.0
- Pandas
- Numpy
  
# Framework
![figure](https://github.com/kangyuseung/CGAN-based-Route-Recommendation/assets/155530934/6ee748f6-f03f-4cc2-bd1b-b5da56fcd4ad)
