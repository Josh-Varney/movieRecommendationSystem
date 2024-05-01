import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing

def printHistogram(train_movies_df):
    """
    Plots histograms for numerical variables in the movie dataset.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']].hist(bins=20, figsize=(12, 8))
    plt.suptitle('Histograms of Numerical Variables', y=0.95)
    plt.show()
    

def printPairPlot(train_movies_df):
    """
    Plots a pairplot to visualize relationships between pairs of numerical variables in the movie dataset.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    
    Returns:
    - None
    """
    sns.pairplot(train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']]) 
    plt.suptitle('Pairplot of Numerical Variables')
    plt.show()
    

def printCorrelationMatrix(train_movies_df):
    """
    Calculates correlation matrix and visualizes it using a heatmap.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    
    Returns:
    - None
    """
    correlation_matrix = train_movies_df[['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']].corr()       
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Matrix')
    plt.show()
    

def printBoxPlot(train_movies_df, train_credits_df):
    """
    Plots a boxplot to identify outliers in the 'runtime' variable of the movie dataset.
    
    Parameters:
    - train_movies_df (DataFrame): DataFrame containing movie data.
    - train_credits_df (DataFrame): DataFrame containing movie credits data.
    
    Returns:
    - None
    """
    sns.boxplot(train_movies_df['runtime'])
    plt.show()

if __name__ == '__main__':
    movies = preprocessing.preprocessTMDSet()
    
    printCorrelationMatrix(movies)
