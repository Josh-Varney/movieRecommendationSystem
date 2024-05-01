import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import preprocessing  

# calculating weighted averages and scores for movies
def calculateWeights(movies):
    """
    Calculate weighted averages and scores for movies.

    Args:
    - movies (DataFrame): Preprocessed movie dataset.

    Returns:
    - DataFrame: Movies DataFrame with weighted averages and scores.
    """
    
    # weighted average calculation 
    R = movies['vote_average']
    v = movies['vote_count']
    m = movies['vote_count'].quantile(0.9)  # 90% quantiles of the dataset
    C = movies['vote_average'].mean()

    movies['weighted_average'] = (R*v + C*m)/(v+m)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(movies[['popularity', 'weighted_average']])
    weighted_df = pd.DataFrame(scaled, columns=['popularity', 'weighted_average'])

    weighted_df.index = movies['original_title']

    # strike balance between popularity and weighted_average
    weighted_df['score'] = weighted_df['weighted_average']*0.4 + weighted_df['popularity'].astype('float64')*0.6
    
    return movies, weighted_df


# tag creation for search
def createTags(movies):
    """
    Create tags for movies based on their features.

    Args:
    - movies (DataFrame): Preprocessed movie dataset.

    Returns:
    - DataFrame: Movies DataFrame with added 'tags' column.
    """
    # null value check
    if movies[['title', 'tagline', 'cast', 'crew']].isnull().values.any():
        print("Warning: Null values found in the dataset.")

    # string conversion
    movies['cast'] = movies['cast'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
    movies['crew'] = movies['crew'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))

    # fill NaN values 
    movies['tagline'] = movies['tagline'].fillna('')

    # convert to string
    movies = movies.astype(str)

    # column concatenation
    movies['concatenated_text'] = (movies['title'] + ' ' + 
                                movies['tagline'] + ' ' + 
                                movies['cast'] + ' ' + 
                                movies['crew'])

    # extract keywords
    vectorizer = CountVectorizer(stop_words='english')

    # activate vectorizer 
    keywords_matrix = vectorizer.fit_transform(movies['concatenated_text'])

    # get feature keywords
    keywords = vectorizer.get_feature_names_out()

    # keywords matrix to DF
    keywords_df = pd.DataFrame(keywords_matrix.toarray(), columns=keywords)

    # add tags column containing keywords separated by commas
    movies['tags'] = keywords_df.apply(lambda row: ', '.join(keywords[row > 0]), axis=1)

    # 'concatenated_text' column drop
    movies.drop(columns=['concatenated_text'], inplace=True)

    return movies


# filter by keywords
def filterMovies(keyword, movies):
    """
    Filter movies based on a keyword.

    Args:
    - keyword (str): Keyword to filter movies.
    - movies (DataFrame): Preprocessed movie dataset.

    Returns:
    - DataFrame: Top-rated movies containing the keyword.
    """
    # filter movies by keyword in the 'tags' column
    keyword_movies = movies[movies['tags'].str.contains(keyword, case=False)]

    if keyword_movies.empty:
        print("No movies found matching the keyword.")
    else:
        # ['vote_average'] in descending order
        top_rated_keyword_movies = keyword_movies.sort_values(by='vote_average', ascending=False).head(10)
        top_rated_keyword_movies = top_rated_keyword_movies[['title', 'vote_average', 'tags']]
        return top_rated_keyword_movies
    
    return []


# function for calculating movie recommendations based on search keywords using Euclidean distance
def calculateThroughTags(preprocessed_movies, search_sentence):
    """
    Calculate movie recommendations based on search keywords using Euclidean distance.

    Args:
    - preprocessed_movies (DataFrame): Preprocessed movie dataset.
    - search_sentence (str): Search keyword or sentence.

    Returns:
    - DataFrame: Recommended movies matching the search keyword.
    """
    movies, weights = calculateWeights(preprocessed_movies)
    
    weights = weights.sort_values(by='score', ascending=False)
    
    movies = createTags(movies=movies)
    
    movies = movies.dropna(subset=['tags'])
    
    # TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    
    # euclidean distances and sorting
    euclidean_dist = euclidean_distances(tfidf_matrix, tfidf_matrix)
    
    search_idx = movies[movies['tags'].str.contains(search_sentence, case=False)].index[0]
    
    similar_movies_idx = euclidean_dist[search_idx].argsort()[1:11]  # Exclude the movie itself
    recommended_movies = movies.iloc[similar_movies_idx]
    
    print(recommended_movies)
    
    return recommended_movies


def tagRecommendation(search_word):
    """
    Initiate the Tag Filtering.

    Returns:
    - List: Top-rated movies based on Cosine Tag Analysis
    """
    # preprocess movie dataset
    movies = preprocessing.preprocessTMDSet()
    
    recommendations_df = calculateThroughTags(movies, search_word)
    
    # convert to 1D
    titles_1d_array = recommendations_df['title'].tolist()
    
    return titles_1d_array


def evaluate_tag_recommendation(recommended_movies, actual_keyword):
    """
    Evaluate the recommendation system based on keyword tags.

    Args:
    - recommended_movies (DataFrame): DataFrame of recommended movies.
    - actual_keywords (str): Actual keyword.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # recommended movie's keywords contain at least one actual keyword check
    for movie_name in recommended_movies['title']:
        boolval = False
        record = recommended_movies[recommended_movies['title'] == movie_name]
        for tag in record['tags']:
            if actual_keyword in tag:
                true_positives += 1
                boolval = True
            
        if not boolval:
            false_negatives += 1
            
    # false positives calc
    false_positives = len(recommended_movies) - true_positives

    # evaluation metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    accuracy = true_positives / len(recommended_movies) if len(recommended_movies) > 0 else 0

    # dict of evaluation metrics
    evaluation_metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Accuracy': accuracy
    }
    
    return evaluation_metrics

# run evaluation
def EvalRecommendation(search_word):
    """
    Initiate the Tag Filtering.

    Returns:
    - List: Top-rated movies based on Cosine Tag Analysis
    """
    # preprocess movie dataset
    movies = preprocessing.preprocessTMDSet()
    
    recommendations_df = calculateThroughTags(movies, search_word)
    
    return recommendations_df


if __name__ == '__main__':
    # keywords for evaluation
    keywords = ['action', 'romance', 'faith', 'beauvois', 'face', 'gods', 'greatest', 'lambert', 'armstrong', 'captive', 'david', 'faith', 'jameson', 'jerry', 'hero', 'immortals', 'danny', 'donner', 'expect', 'faces']
    
    # average metrics
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_accuracy = 0
    
    for count, key in enumerate(keywords, 1):
        print(f'Epoch {count}/{len(keywords)}: {key}')
        recommendations = EvalRecommendation(key)  
        metrics = evaluate_tag_recommendation(recommendations, key) 
        
        print(f'Recommended movies for keyword "{key}":')
        print(recommendations[['title', 'vote_average', 'tags']])
            
        avg_precision += metrics['Precision']
        avg_recall += metrics['Recall']
        avg_f1_score += metrics['F1 Score']
        avg_accuracy += metrics['Accuracy']
    
    avg_precision /= len(keywords)
    avg_recall /= len(keywords)
    avg_f1_score /= len(keywords)
    avg_accuracy /= len(keywords)
    
    print('Average Precision: ', avg_precision)
    print('Average Recall: ', avg_recall)
    print('Average F1 Score: ', avg_f1_score)
    print('Average Accuracy: ', avg_accuracy)
