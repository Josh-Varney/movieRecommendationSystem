import time
import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold


def preprocess_movies(movies):
    """
    Preprocess the movie data.

    Args:
    - movies (DF): DF containing movie data.

    Returns:
    - DF: Preprocessed movie DF.
    """
    
    movies['overview'] = movies['overview'].astype(str).fillna('')
    return movies

# tfidf calc
def calculate_tfidf_matrix(movies):
    """
    Calculate the TF-IDF matrix for movie overviews.

    Args:
    - movies (DF): Preprocessed DF containing movie data.

    Returns:
    - (TfidfVectorizer, scipy.sparse.csr_matrix): Tuple containing the TF-IDF vectorizer and TF-IDF matrix.
    """
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    return tfidf, tfidf_matrix

# cosine similarity calc
def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculate cosine similarity matrix.

    Args:
    - tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix for movie overviews.

    Returns:
    - numpy.ndarray: Cosine similarity matrix.
    """
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# recommendation obtain
def get_recommendations(title, indices, cosine_sim, movies):
    """
    Get movie recommendations based on cosine similarity.

    Args:
    - title (str): Title of the movie to find similar movies.
    - indices (pandas.Series): Series containing indices of movies.
    - cosine_sim (numpy.ndarray): Cosine similarity matrix.
    - movies (DF): DF containing movie data.

    Returns:
    - list: List of recommended movie titles.
    """
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].values
    except KeyError:
        # Return an empty array if the key does not exist in the index (movie not found)
        return []

# initialise recommendation process
def calculate_through_cosine(movies, search_sentence):
    """
    Calculate movie recommendations based on cosine similarity.

    Args:
    - movies (DF): DF containing movie data.
    - search_sentence (str): The search sentence to find similar movies.

    Returns:
    - recommended_movies (list): List of recommended movie titles.
    """
    movies = preprocess_movies(movies)
    tfidf, tfidf_matrix = calculate_tfidf_matrix(movies)
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)

    # create indices series
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # get recommendations
    recommendations = get_recommendations(search_sentence, indices, cosine_sim, movies)
    
    return recommendations

# user runs to start normal process
def cosineRecommendation(search_word):
    """
    Initiate the cosine recommendation system

    Returns:
    - list : List of recommended movie titles.
    """
    # preprocess obtain
    movies = preprocessing.preprocessTMDSet()
    
    # initialise the process
    recommendations = calculate_through_cosine(movies=movies, search_sentence=search_word)
    
    if len(recommendations) > 0:
        titles_1d_array = recommendations.tolist()
        return titles_1d_array
    else:
        print("No recommendations found.")
        return []

# runs evaluative statistic calculations 
def evaluate_recommendation(recommended, actual_title, actual_list, k=5):
    """
    Evaluate the recommendation system using k-fold cross-validation.

    Args:
    - recommended (list): List of recommended movie titles.
    - actual_title (str): Title of the actual movie.
    - actual_list (list): List of actual movie titles.
    - k (int): Number of folds for k-fold cross-validation.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    kf = KFold(n_splits=k)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    count = 0
    start_time = time.time()
    for train_index, test_index in kf.split(recommended):
        print(f'Epoch: {count+1} / {k}')
        train_data = [recommended[i] for i in train_index]
        test_data = [recommended[i] for i in test_index]
        
        for film in test_data:
            boolval = False
            # if the actual movie is in the recommended list check
            if actual_title in cosineRecommendation(film) and not boolval:
                true_positives += 1
                boolval = True
            elif film in actual_list or not boolval:
                false_positives += 1
            else:
                false_negatives += 1
        count += 1
    end_time = time.time()
    print(f'Test Time: {end_time-start_time}')
    
    # calculation of metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(recommended) if recommended else 0
    
    evaluation_metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy
    }
    
    return evaluation_metrics

if __name__ == '__main__':
    # actual film and finds similarities
    title = "The Wolf of Wall Street"
    
    # One Iteration Time 
    start_time = time.time()
    recommendations = cosineRecommendation(title)
    end_time = time.time()
    print(f'One Iteration Time: {end_time-start_time}')

    # popular films
    actual_movies = ["The Great Gatsby", "The Revenant", "Inception"]

    # evaluate using k-fold cross-validation
    evaluation_metrics = evaluate_recommendation(recommended=recommendations,
                                                  actual_title=title,
                                                  actual_list=actual_movies,
                                                  k=5)

    print("Evaluation Metrics:")
    print(evaluation_metrics)
    