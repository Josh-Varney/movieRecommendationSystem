import ast
import pandas as pd

def preprocessTMDSet():
    """
    Preprocesses the TMDb movie dataset by performing the following steps:
    1. Loads the 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' datasets.
    2. Merges the datasets based on the movie title.
    3. Imputes missing values for selected columns with -1.
    4. Converts the release date to the year.
    5. Converts stringified lists to actual lists for genres, keywords, cast, production companies, and spoken languages.
    6. Extracts the director from the crew list.
    7. Removes movies that are not released.

    Parameters:
    - None

    Returns:
    - DataFrame: Preprocessed movie dataset.
    """
    # Load datasets
    train_movies_df = pd.read_csv("C:\\Users\\Jrv12\\Desktop\\Recommendation System\\recommendationSystem\\datasets\\tmdb_5000_movies.csv")
    train_credits_df = pd.read_csv("C:\\Users\\Jrv12\\Desktop\\Recommendation System\\recommendationSystem\\datasets\\tmdb_5000_credits.csv")

    # Merge datasets
    movies = train_movies_df.merge(train_credits_df, on="title")
    
    # Imputation of Missing Values
    def fillMissingValues():
        movies.fillna({"homepage": -1, "overview": -1, "release_date": -1, "runtime": -1, "tagline": -1}, inplace=True)
    fillMissingValues()
    
    # Convert release_date to year
    movies["release_date"] = movies["release_date"].apply(lambda x: [str(x)[:4]])
    
    # Convert stringified lists to lists
    def convert_to_name(obj):
        return [i["name"] for i in ast.literal_eval(obj)]
    movies["genres"] = movies["genres"].apply(convert_to_name)
    movies["keywords"] = movies["keywords"].apply(convert_to_name)
    movies["cast"] = movies["cast"].apply(lambda x: convert_to_name(x)[:3])
    movies["production_companies"] = movies["production_companies"].apply(convert_to_name)
    movies["spoken_languages"] = movies["spoken_languages"].apply(lambda x: convert_to_name(x)[:3])
    
    # Extract director from crew
    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return [i["name"]]
        return []
    movies["crew"] = movies["crew"].apply(fetch_director)
    
    # Remove unreleased movies
    movies = movies[movies['status'] == 'Released']
    
    return movies

if __name__ == '__main__':
    preprocessTMDSet()
    
# Potential Outliers in Dataset Information:
# Numerical Data:
# - Budget, Revenue, Runtime, Popularity, Vote Average, Vote Count
#   Outliers can be identified using Z-Score or IQR. For example, a movie with an unusually high or low budget compared to others may be considered an outlier.

# Categorical Data:
# - Genres, Keywords
#   Rare genres or keywords, such as "Experimental", need to be identified as outliers.

# Binary Data:
# - Homepage
#   Movies with an official website or not. Outliers may be well-developed or poorly developed compared to others.

# Categorical Data:
# - Original Language, Production Companies, Production Countries, Spoken Languages
#   Movies produced in rare languages or countries, or those associated with less-known production companies may be considered outliers.

# Textual Data:
# - Overview, Tagline
#   Movies with exceptionally long or short overviews or taglines compared to others may be outliers.

# Impact on Analysis:
# - Outliers may significantly distort user preferences and recommendations.
# - Removal of outliers may increase the validity of the analysis by reducing their influence on the results.
