import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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
        # Return an empty array if the key does not exist in the index
        return []


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

    # Create indices series
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # Get recommendations
    recommendations = get_recommendations(search_sentence, indices, cosine_sim, movies)
    
    return recommendations


def cosineRecommendation(search_word):
    """
    Initiate the cosine recommendation system

    Returns:
    - list : List of recommended movie titles.
    """
    # preprocess obtain
    movies = preprocessing.preprocessTMDSet()
    
    # Consine similarity calc
    recommendations = calculate_through_cosine(movies=movies, search_sentence=search_word)
    
    if len(recommendations) > 0:
        titles_1d_array = recommendations.tolist()
        return titles_1d_array
    else:
        print("No recommendations found.")
        return []
    
def evaluate_recommendation(recommended, actual_title, actual_list):
    """
    Evaluate the recommendation system using metrics.

    Args:
    - recommended (list): List of recommended movie titles.
    - actual_title (str): Title of the actual movie.
    - actual_list (list): List of actual movie titles.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    print('This make take several minutes')
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    count = 0
    for film in recommended:
        boolval = False
        count += 1
        
        print(f'Epoch {count}/{len(recommended)}: ')
        
        new_recommendations = cosineRecommendation(film)
        print(new_recommendations)
        
        # if the actual movie is in the recommended list check
        if actual_title in new_recommendations and not boolval:
            true_positives += 1
            boolval = True
        elif film in actual_list or not boolval:
            false_positives += 1
        else:
            false_negatives += 1
    
    # calc metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    accuracy = true_positives / len(recommended) if recommended else 0
    
    # dict evaluation metrics
    evaluation_metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy
    }
    
    print(evaluation_metrics)
    
    return evaluation_metrics

if __name__ == '__main__':
    # Actual film and finds similarities
    title = "The Wolf of Wall Street"
    recommendations = cosineRecommendation(title)
    
    # Popular films 
    actual_movies = ["The Great Gatsby", "The Revenant", "Inception"]
    
    # Evaluate each recommended movie
    evaluation_metrics = evaluate_recommendation(recommendations, title, actual_movies)
    