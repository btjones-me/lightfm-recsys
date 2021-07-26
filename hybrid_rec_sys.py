from collections import defaultdict
import csv
import numpy
import random

import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings("ignore")

import tensorrec

import logging
# logging.getLogger().setLevel(logging.INFO)

# Open and read in the rating file
# NOTE: This expects the ratings.csv file to be in the same folder as this Python file.
random.seed(200)
ratings_path = 'data/interactions.csv'
print('Loading ratings')
with open(ratings_path, 'r') as ratings_file:
    ratings_file_reader = csv.reader(ratings_file)
    raw_ratings = list(ratings_file_reader)
    raw_ratings_header = raw_ratings.pop(0)

# Iterate through the input to map Data IDs to new internal IDs
# The new internal IDs will be created by the defaultdict on insertion
ratings_data_to_internal_user_ids = defaultdict(lambda : len(ratings_data_to_internal_user_ids))
ratings_data_to_internal_item_ids = defaultdict(lambda : len(ratings_data_to_internal_item_ids))

for row in raw_ratings:
    row[0] = ratings_data_to_internal_user_ids[row[0]]
    row[1] = ratings_data_to_internal_item_ids[row[1]]
    row[2] = float(row[2])
n_users = len(ratings_data_to_internal_user_ids)
n_items = len(ratings_data_to_internal_item_ids)

# Look at an example raw rating
print('Raw ratings/interactions example:\n{}\n{}'.format(raw_ratings_header, raw_ratings[0]))

# Shuffle the ratings and split them in to train/test sets 80%/20%
random.shuffle(raw_ratings) # Shuffles the list in-place
cutoff = int(0.8*len(raw_ratings))
train_ratings = raw_ratings[:cutoff]
test_ratings = raw_ratings[cutoff:]
print('{} train ratings,{} test ratings'.format(len(train_ratings), len(test_ratings)))

# This method converts a list of (user, item, time) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions):
    users_column, items_column, ratings_column = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                             shape=(n_users, n_items))

# Create sparse matrices of interaction data
sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings)

# Construct indicator features for users and items
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)

# Build a matrix factorization collaborative filter model
cf_model = tensorrec.TensorRec(n_components=5)

# Fit the collaborative filter model
print("Training collaborative filter")
cf_model.fit(interactions=sparse_train_ratings,
             user_features=user_indicator_features,
             item_features=item_indicator_features)

# Create sets of train/test interactions that are only ratings >= 1.0
#sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 1.0)
#sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 1.0)


# This method consumes item ranks for each user and prints out recall@10 train/test metrics
def check_results(ranks):
    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_train_ratings,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_test_ratings,
        predicted_ranks=ranks,
        k=10
    ).mean()
    print("Recall at 10: Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
                                                            test_recall_at_10))


# Check the results of the MF CF model
print("Matrix factorization collaborative filter:")
predicted_ranks = cf_model.predict_rank(user_features=user_indicator_features,
                                        item_features=item_indicator_features)
check_results(predicted_ranks)

# Let's try a new loss function: WMRB
print("Training collaborative filter with WMRB loss")
ranking_cf_model = tensorrec.TensorRec(n_components=5,
                                       loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_model.fit(interactions=sparse_train_ratings,
                     user_features=user_indicator_features,
                     item_features=item_indicator_features,
                     n_sampled_items=int(n_items*0.5))

# Check the results of the WMRB MF CF model
print("WMRB matrix factorization collaborative filter:")
predicted_ranks = ranking_cf_model.predict_rank(user_features=user_indicator_features,
                                                item_features=item_indicator_features)
check_results(predicted_ranks)

# To improve the recommendations, lets read in
print('Loading item metadata')
with open('data/item_features_processed.csv', 'r') as item_file:
    items_file_reader = csv.reader(item_file)
    raw_item_metadata = list(items_file_reader)
    raw_item_metadata_header = raw_item_metadata.pop(0)

# Map the item IDs to our internal IDs and keep track of the items and names
item_vars_by_internal_id = {}
item_name_by_internal_id = {}
for row in raw_item_metadata:
    row[0] = ratings_data_to_internal_item_ids[row[1]]  # Map to IDs
    item_vars_by_internal_id[row[0]] = row[2:]
    item_name_by_internal_id[row[0]] = row[1]

# Look at an example item metadata row
print("Raw metadata example:\n{}\n{}".format(raw_item_metadata_header,
                                             raw_item_metadata[0]))

# Build a list of vars where the index is the internal ID and
# the value is a list of [var, var, ...]
item_vars = [item_vars_by_internal_id[internal_id] for internal_id in range(n_items)]

# Transform the genres into binarized labels using scikit's MultiLabelBinarizer
#movie_genre_features = MultiLabelBinarizer().fit_transform(movie_genres)
item_vars_arr = np.array(item_vars)
item_vars_size = item_vars_arr.shape[1]
# print("Binarized genres example for movie {}:\n{}".format(movie_titles_by_internal_id[0],
#                                                          movie_genre_features[0]))

# Coerce the movie genre features to a sparse matrix, which TensorRec expects
item_features = sparse.coo_matrix(item_vars_arr)

# Fit a content-based model using the genres as item features
print("Training content-based recommender")
content_model = tensorrec.TensorRec(
    n_components=item_vars_size,
    item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph()
)
content_model.fit(interactions=sparse_train_ratings,
                  user_features=user_indicator_features,
                  item_features=item_features,
                  n_sampled_items=int(n_items * 0.5))

# Check the results of the content-based model
print("Content-based recommender:")
predicted_ranks = content_model.predict_rank(user_features=user_indicator_features,
                                             item_features=item_features)
check_results(predicted_ranks)

# Try concatenating the vars on to the indicator features for a hybrid recommender system
full_item_features = sparse.hstack([item_indicator_features, item_features.astype(float)])

print("Training hybrid recommender")
hybrid_model = tensorrec.TensorRec(
    n_components=5,
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph()
)
hybrid_model.fit(interactions=sparse_train_ratings,
                 user_features=user_indicator_features,
                 item_features=full_item_features,
                 n_sampled_items=int(n_items * .5))

print("Hybrid recommender:")
predicted_ranks = hybrid_model.predict_rank(user_features=user_indicator_features,
                                            item_features=full_item_features)
check_results(predicted_ranks)

