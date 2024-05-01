import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import preprocessing 
import content_euclidean_tag_based


def recommend_movies_with_tags(preprocessed_movies, search_sentence):
    """
    Recommend movies based on search keywords using both Euclidean distance and Cosine similarity.

    Args:
    - preprocessed_movies (DataFrame): Preprocessed movie dataset.
    - search_sentence (str): Search keyword or sentence.

    Returns:
    - DataFrame: Recommended movies matching the search keyword.
    """
    movies, weights = content_euclidean_tag_based.calculateWeights(preprocessed_movies)
    
    # sort by weight score
    weights = weights.sort_values(by='score', ascending=False)
    
    movies = content_euclidean_tag_based.createTags(movies=movies)
    
    # drop NaN in tags cplumn
    movies = movies.dropna(subset=['tags'])
    
    # TF-IDF matrix calc
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    
    # euclidean distances calc
    euclidean_dist = euclidean_distances(tfidf_matrix, tfidf_matrix)
    
    # cosine similarities calc
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # index of the search keyword search
    search_idx = movies[movies['tags'].str.contains(search_sentence, case=False)].index[0]
    
    # euclidean and cosine sort and excludes the movie itself
    euclidean_similar_movies_idx = euclidean_dist[search_idx].argsort()[1:11]  
    euclidean_recommendations = movies.iloc[euclidean_similar_movies_idx]
    
    cosine_similar_movies_idx = cosine_sim[search_idx].argsort()[::-1][1:11]  
    cosine_recommendations = movies.iloc[cosine_similar_movies_idx]
    
    # combine recommendations from both methods
    combined_recommendations = pd.concat([euclidean_recommendations, cosine_recommendations]).drop_duplicates().reset_index(drop=True)
    
    return combined_recommendations


    
def evaluate_euclidean_cosine_recommendation(recommended_movies, actual_keyword):
    """
    Evaluate the recommendation system based on keyword tags.

    Args:
    - recommended_movies (DataFrame): Recommended movies DataFrame.
    - actual_keyword (str): Actual keyword.

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
            
    # false positive calculation
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


def ui_interactions(search_word):
    movies = preprocessing.preprocessTMDSet()
    recommendations_df = recommend_movies_with_tags(search_sentence=search_word, preprocessed_movies=movies)
    
    titles_1d_array = recommendations_df['title'].tolist()
    return titles_1d_array


def main():
    # preprocessed dataset
    movies = preprocessing.preprocessTMDSet()
    keywords = ['action', 'romance', 'faith', 'beauvois', 'face', 'gods', 'greatest', 'lambert', 'armstrong', 'captive', 'david', 'faith', 'jameson', 'jerry', 'hero', 'immortals', 'danny', 'donner', 'expect', 'faces']
    
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_accuracy = 0
    
    for count, key in enumerate(keywords, 1):
        print(f'Epoch {count}/{len(keywords)}: {key}')
        recommendations = recommend_movies_with_tags(movies, key)  
        metrics = evaluate_euclidean_cosine_recommendation(recommendations, key)  
        
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


if __name__ == '__main__':
    main()
 
