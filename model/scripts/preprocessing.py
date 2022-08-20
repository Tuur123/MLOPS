
import pandas as pd
import numpy as np

def read_data():
    dataset = pd.read_csv('../data/MovieLens/ratings.csv',sep=',',header=None, engine='python', skiprows=1,
                     names=["User", "Movie", "Rating", "Timestamp"])

    dataset['Rating'] = dataset['Rating'] / 5.0
    movieratings_dataframe = dataset.pivot(index='User',columns='Movie',values='Rating').fillna(0)
    movieratings_dataframe = movieratings_dataframe.reindex(movieratings_dataframe.columns.union(np.arange(1, max(dataset['Movie']))), axis=1, fill_value=0.0)
    return movieratings_dataframe



def split_data(data):
    movieratings = np.asarray(data)

    test_ratings_per_user = 10  # The amount of ratings of one user that we will copy to the Test-set
    X_train = movieratings.copy() # Copy the whole list
    X_test = np.zeros(movieratings.shape) # Make a list with only zeros
    y_train = movieratings.copy() # Copy the whole list
    y_test = movieratings.copy() # Copy the whole list

    ratedlist = [] # This will contain all the movieratings in our test_set so we can check it later
    for i in range(0, movieratings.shape[0]): # For every user, we have to set a few movieratings to 0
        ratedmovies = np.where(movieratings[i,:] > 0.0)[0] # Get a list of movies with a rating greater than 0
        ratedlist.append(ratedmovies)
        index = np.random.choice(len(ratedmovies), test_ratings_per_user, replace=False)
        
        X_train[i,ratedmovies[index]] = 0 # Set rating to 0 in training set
        X_test[i,ratedmovies[index]] = movieratings[i,ratedmovies[index]].copy() # Copy the value of the rating to the test set

    return X_train, X_test, y_train, y_test, ratedlist


def save_data(X_train, X_test, y_train, y_test, ratedlist, path):
    np.save(path + "X_train.npy", X_train)
    np.save(path + "y_train.npy", y_train)
    np.save(path + "X_test.npy", X_test)
    np.save(path + "y_test.npy", y_test)
    np.save(path + "ratedlist.npy", np.asarray(ratedlist))


data = read_data()

X_train, X_test, y_train, y_test, ratedlist = split_data(data)

save_data(X_train, X_test, y_train, y_test, ratedlist, "../data/preprocessed_data/")

