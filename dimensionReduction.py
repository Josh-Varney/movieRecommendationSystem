import numpy as np
import pandas as pd
import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

def identifyOutliers(std_limit, train_movies_df):
    """
    Identifies outliers in the movie dataset using z-score or Interquartile Range (IQR) methods.
    
    Parameters:
    - std_limit (float): Standard deviation limit for identifying outliers.
    - train_movies_df (DataFrame): DataFrame containing movie data.
    
    Returns:
    - DataFrame: DataFrame containing the outliers.
    """
    # Exclude non-numeric columns and the 'id' column
    numeric_columns = train_movies_df.drop(columns=['id', 'popularity']).select_dtypes(include='number')
    
    upper_limit = numeric_columns.mean() + std_limit * numeric_columns.std()
    lower_limit = numeric_columns.mean() - std_limit * numeric_columns.std()
    
    # Align DataFrame
    numeric_columns, _ = numeric_columns.align(upper_limit, axis=1, copy=False)
    
    outliers = (numeric_columns > upper_limit) | (numeric_columns < lower_limit)
    
    return train_movies_df[outliers.any(axis=1)]


def removeOutliers(std_limit, train_movies_df):
    """
    Removes outliers from the movie dataset using z-score or Interquartile Range (IQR) methods.
    
    Parameters:
    - std_limit (float): Standard deviation limit for identifying outliers.
    - train_movies_df (DataFrame): DataFrame containing movie data.
    
    Returns:
    - DataFrame: DataFrame with outliers removed.
    """
    outliers_df = identifyOutliers(std_limit, train_movies_df)
    cleaned_df = train_movies_df.drop(outliers_df.index)
    
    return cleaned_df


def correlationReduction(train_movies_df, threshold):
    """
    Reduces dimensionality by identifying highly correlated features and removing them.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    - threshold (float): Threshold for correlation coefficient.
    
    Returns:
    - set: Set of column names representing highly correlated features.
    """
    col_corr = set()
    corr_matrix = train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']].corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


def performPCA(train_movies_df, train_credits_df):
    """
    Performs Principal Component Analysis (PCA) on the movie dataset to reduce its dimensionality.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    - train_credits_df (DataFrame): DataFrame containing movie credits data.
    
    Outputs:
    - Explained variance ratio plot.
    - Loadings for the first principal component.
    
    Returns:
    - None
    """
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']])

    pca = PCA() 
    pca.fit(scaled_data)

    variance_ratio = pca.explained_variance_ratio_

    explained_variance = np.insert(variance_ratio, 0, 0)
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    df_explained_variance = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    })

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker='o', linestyle='-')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    first_pc_loadings = pca.components_[1]

    loadings_df = pd.DataFrame({'Variable': ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count'],
                                'Loading': first_pc_loadings})

    loadings_df['Abs_Loading'] = loadings_df['Loading'].abs()
    loadings_df = loadings_df.sort_values(by='Abs_Loading', ascending=False)

    print("Loadings for the First Principal Component:")
    print(loadings_df)

        
if __name__ == '__main__':
    movies = preprocessing.preprocessTMDSet()
    
    # Run a function to test any difference in the dataset