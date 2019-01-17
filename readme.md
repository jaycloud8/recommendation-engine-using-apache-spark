# Recommendation Engine

A recommendation engine can make predictions for unknown user-item associations based on known user-item association.

User-item association is user's affinity to item.

Recommendation engine builds a prediction model based on known user-item associations (Training data) 
and then make predictions for unknown user-item associations (Test Data).

## Collaborative Filtering Technique
Collaborative Filtering based recommender engines use matrix factorization model typically 
(Alternating Least Square method) to fill the missing entries of user-item association. 

Item ratings are influenced by a set of factors, such as user’s choices and interests, called Latent Factors.

More the latent factors, better will be the predictions.

The standard approach to matrix factorization based collaborative filtering treats the entries 
in the user-item matrix as `explicit preferences` given by the user to the item, for example, 
users giving ratings to movies.

It is common in many real-world use cases to only have access to `implicit feedback` 
(e.g. views, clicks, purchases, likes, shares etc.).

In the following example, we load ratings data from the MovieLens dataset, 
each row consisting of a user, a movie, a rating and a timestamp. 
We then train an ALS model which assumes, by default, that the ratings are explicit (implicitPrefs is false). 
We evaluate the recommendation model by measuring the `root-mean-square error` of rating prediction.