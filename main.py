# from datetime import datetime
import pandas as pd
import numpy as np

import seaborn as sns
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# import xgboost as xgb
# from surprise import Reader, Dataset
# from surprise import BaselineOnly
# from surprise import KNNBaseline
# from surprise import SVD
# from surprise import SVDpp
# from surprise.model_selection import GridSearchCV


def load_data():
    df = pd.read_csv(
        "./datasets/ratings.csv",
        sep=",",
        # names=["book_id","user_id","rating"],
        encoding = "latin",
        low_memory = False
    )
    data = pd.DataFrame(df)
    id = [i for i in range(len(df))]
    data['id'] = id
    return df


book_rating_df = load_data()[1:]
book_rating_df.duplicated(["book_id","user_id","rating"]).sum()

print(book_rating_df.head())
print(type(book_rating_df.head()))


split_value = int(len(book_rating_df) * 0.80)
train_data = book_rating_df[:split_value]
test_data = book_rating_df[split_value:]

print(len(train_data))
print(len(test_data))

plt.figure(figsize = (12, 8))
ax = sns.countplot(x="rating", data=train_data)

ax.set_yticklabels([num for num in ax.get_yticks()])

plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
# plt.show()

def get_user_item_sparse_matrix(df):
    sparse_data = sparse.csr_matrix( (df.rating, (df.user_id, df.book_id )) )
    return sparse_data

train_sparse_data = get_user_item_sparse_matrix(train_data)
test_sparse_data = get_user_item_sparse_matrix(test_data)
global_average_rating = train_sparse_data.sum() / train_sparse_data.count_nonzero()
print("Global Average Rating: {}".format(global_average_rating))
# print(train_sparse_data)

def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1
    no_of_ratings = (sparse_matrix != 0).sum(axis = ax).A1
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings

average_rating_user = get_average_rating(train_sparse_data, True)

avg_rating_book = get_average_rating(train_sparse_data, False)

total_users = len(np.unique(book_rating_df["user_id"]))
train_users = len(average_rating_user)
uncommonUsers = total_users - train_users

print("Total no. of Users = {}".format(total_users))
print("No. of Users in train data= {}".format(train_users))
print("No. of Users not present in train data = {}({}%)".format(uncommonUsers, np.round((uncommonUsers/total_users)*100), 2))

total_book = len(np.unique(book_rating_df["book_id"]))
train_book = len(avg_rating_book)
uncommonBooks = total_book - train_book

print("Total no. of Books = {}".format(total_book))
print("No. of Books in train data= {}".format(train_book))
print("No. of Books not present in train data = {}({}%)".format(uncommonBooks, np.round((uncommonBooks/total_book)*100), 2))

def compute_user_similarity(sparse_matrix, limit=100):
    row_index, col_index = sparse_matrix.nonzero()
    rows = np.unique(row_index)
    similar_arr = np.zeros(67100).reshape(671,100)

    for row in rows[:limit]:
        sim = cosine_similarity(sparse_matrix.getrow(row), train_sparse_data).ravel()
        similar_indices = sim.argsort()[-limit:]
        similar = sim[similar_indices]
        similar_arr[row] = similar

    return similar_arr

similar_user_matrix = compute_user_similarity(train_sparse_data, 100)

similar_user_matrix[0]

book_titles_df = pd.read_csv("./datasets/books.csv",
                                sep=",",
                                # names=['ISBN', 'Year_Of_Publication', 'Book_Title', 'Publisher'],
                                encoding="latin",
                                low_memory=False)
print(book_titles_df.head())

def compute_book_similarity_count(sparse_matrix, book_titles_df, book_id):
    similarity = cosine_similarity(sparse_matrix.T, dense_output = False)
    no_of_similar_books = book_titles_df.loc[book_id]['original_title'], similarity[book_id].toarray()
    return no_of_similar_books

name, similar_books = compute_book_similarity_count(train_sparse_data, book_titles_df, 10)
print("Similar Books = {}".format(name))
similar_books_id = [(similar_books[0][i], i) for i in range(len(similar_books[0]))]
print(sorted(similar_books_id)[-10:] )

def get_sample_sparse_matrix(sparse_matrix, no_of_users, no_of_books):
    users, books, ratings = sparse.find(sparse_matrix)
    uniq_users = np.unique(users)
    uniq_books = np.unique(books)
    np.random.seed(15)
    user = np.random.choice(uniq_users, no_of_users, replace = False)
    book = np.random.choice(uniq_books, no_of_books, replace = True)
    mask = np.logical_and(np.isin(users, user), np.isin(books, book))
    sparse_matrix = sparse.csr_matrix((ratings[mask], (users[mask], books[mask])),
                                                     shape = (max(user)+1, max(book)+1))
    return sparse_matrix

train_sample_sparse_matrix = get_sample_sparse_matrix(train_sparse_data, 400, 40)

test_sparse_matrix_matrix = get_sample_sparse_matrix(test_sparse_data, 200, 20)
