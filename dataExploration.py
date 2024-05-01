from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing

def performPCA(train_movies_df):
    """
    Perform Principal Component Analysis (PCA) to identify outliers and reduce dimensionality.

    Args:
        train_movies_df (DataFrame): DataFrame containing movie data.

    Returns:
        DataFrame: DataFrame containing loadings for the first principal component.
    """
    # Standardize the data
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']])

    # Fit PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Explained variance ratio
    variance_ratio = pca.explained_variance_ratio_

    # Calculate cumulative variance
    explained_variance = np.insert(variance_ratio, 0, 0)
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    # Create DataFrame for explained variance
    df_explained_variance = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    })

    # Plot explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker='o', linestyle='-')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    # Extract loadings for the first principal component
    first_pc_loadings = pca.components_[0]

    # Create DataFrame for loadings
    loadings_df = pd.DataFrame({'Variable': ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count'],
                                'Loading': first_pc_loadings})

    # Sort DataFrame by absolute loading values
    loadings_df['Abs_Loading'] = loadings_df['Loading'].abs()
    loadings_df = loadings_df.sort_values(by='Abs_Loading', ascending=False)

    return loadings_df


# Select principle components based on specific objectives and nature of data
if __name__ == '__main__':
    
    # Perform Preprocessing
    movies = preprocessing.preprocessTMDSet()
    
    # Perform PCA analysis
    print('Select a Function to Perform on Data Frame')