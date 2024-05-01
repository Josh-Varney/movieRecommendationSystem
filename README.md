# Dataset Usage

- **tmdb_dataset.zip**: contains the files of both the datasets to use within all of these features as well as in the .ipynb_checkpoints for better visualization.
- **ml-25**: coming soon for collaborative filtering.
- **Please ensure all paths are changed throughout the python files, this should only include `preprocessing.py`**.

# Code Documentation

- This can be review through all `.py` files to understand the functions capabilities and for future expansion purposes.

# User Interface 

This Python script contains basic UI code for eliciting two recommendation features:

1. **Checking the Cosine Button**: Requires the input of an actual film within the dataset, eliciting a similar set of movies based on Cosine Methodologies.

2. **Not Checking the Cosine Button**: Elicits a recommendation search through a keyword, such as 'action' or 'romance'.

3. **This Script Doesn't Contain the Evaluative Scripts**: For both features, these can be found in the scripts `content_cosine_based.py` and `content_tag_based.py`. Load and run the `if __name__ == '__main__' or main()` functions to run a set of evaluative tests.

4. **Future Improvement**: Introducing a user-based system for recommendation would enhance accuracy, precision, and recall, potentially through collaborative and content filtering methods.

5. **Interaction for the user to recommend movies**
 - Please ensure an actual movie is used to produce cosine recommendations when the box is checked
 - Please ensure to enter a relevant tag is used when using euclidean
 - Hybrid model can be used through the main running of the `cosine_euclidean_tag_based.py` has not been yet implemented within the system

# Movie Recommendation System - Data Preprocessing

This repository contains code for preprocessing the TMDb (The Movie Database) movie dataset. The preprocessing steps are aimed at cleaning and organizing the data for further analysis or use in a recommendation system.

`preprocessing.py`

# Movie Dataset Visualization

This script contains functions to visualize the TMDb (The Movie Database) movie dataset. It utilizes various plotting techniques to gain insights into the dataset's numerical variables and relationships between them.

`outerVisualisation.py`

# Movie Dataset Preprocessing and Dimensionality Reduction

This script contains functions for preprocessing the TMDb (The Movie Database) movie dataset and performing dimensionality reduction techniques, such as identifying outliers, removing outliers, correlation reduction, and Principal Component Analysis (PCA).

`dimensionReduction.py`

# Movie Tag-Based Recommendation System

This script implements a recommendation system for movies based on search keywords and evaluates its performance using precision, recall, F1 score, and accuracy metrics.

`content_euclidean_tag_based.py`

# Movie Recommendation System Based on Cosine Similarity

This script implements a movie recommendation system based on cosine similarity between movie overviews. It preprocesses the movie dataset, calculates TF-IDF vectors for movie overviews, computes cosine similarity between movies, and generates recommendations.

`content_cosine_based.py`

By organizing the information into distinct sections with concise descriptions, users can quickly understand the purpose and functionality of each feature in the project.

# Hybid Model

Combination of the above two techniques used to recommend movies to the user.

`cosine_euclidean_tag_based.py`
