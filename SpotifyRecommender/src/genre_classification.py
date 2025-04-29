from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def multi_genre_knn(df, features, target, k):
    #KNN (1 Point)

    X = df[features]
    y = df[target]
    
    #Use the first genre in the list to stratify (or the string itself if it's a single genre)
    y_train_primary = y.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_train_primary
    )

    print(f"After splitting - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    #Extract primary genres for training and testing
    primary_y_train = y_train.apply(lambda x: x[0] if isinstance(x, list) else x)
    primary_y_test = y_test.apply(lambda x: x[0] if isinstance(x, list) else x)
    
    #Train KNN model
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, primary_y_train)
    
    #Predictions
    y_pred = knn.predict(X_test)
    
    #Evaluate accuracy for multiple genres
    custom_accuracy = multi_genre_accuracy_score(y_test, y_pred)
    print(f"Multi-genre accuracy: {custom_accuracy:.4f}")
    
    #Evaluate single-genre for comparison
    test_accuracy = knn.score(X_test, primary_y_test)
    train_accuracy = knn.score(X_train, primary_y_train)

    #Check for underfitting / overfitting
    print(f"Traditional train accuracy: {train_accuracy:.4f}")
    print(f"Traditional test accuracy: {test_accuracy:.4f}")
    
    cm = confusion_matrix(primary_y_test, y_pred)
    
    return knn, X_train, X_test, y_train, y_test, y_pred, cm

def multi_genre_accuracy_score(y_true, y_pred):
    """
    Since some tracks can have multiple genres, we want to consider KNN to make a
    correct prediction if any of those multiple genres match
    """

    correct = 0
    total = len(y_true)
    
    #Pair each genre/list of genres with the singular predicted genre, element to element
    for actual_list, predicted in zip(y_true, y_pred):
        if isinstance(actual_list, str):
            #Makes sure singular genres are lists so it can be compared correctly
            actual_list = [actual_list]
        
        #Check if predicted genre is in the list of true genres
        if predicted in actual_list:
            correct += 1
    
    return correct / total

"""
This confusion matrix is intended for poster visuals, using a sample of the data and subset of the genres.

scaler, pca_model, and knn_model are pretrained models created in pipeline.py
get_primary_genre() is a parameter in this function in order to be called in a Jupiter notebook
top_n: Number of genres to be used in the confuson matrix
"""
def plot_confusion_matrix(filename, processed_df, scaler, pca_model, knn_model, features, get_primary_genre, top_n):
    sampled_df = processed_df.copy()
    #Get the primary genre
    sampled_df['genre_primary'] = sampled_df['genre'].apply(get_primary_genre)

    #Get the n most common genres and only include tracks with those genres
    top_genres = sampled_df['genre_primary'].value_counts().head(top_n).index.tolist()
    filtered_df = sampled_df[sampled_df['genre_primary'].isin(top_genres)]

    sample_size = min(5000, len(filtered_df))
    sampled_data = filtered_df.sample(n=sample_size, random_state=42)

    X = sampled_data[features]
    y = sampled_data['genre_primary']

    #Applying the pretrained scaler, pca, and knn models to the sample
    X_scaled = scaler.transform(X)
    X_pca = pca_model.transform(X_scaled)
    y_pred = knn_model.predict(X_pca)

    #Get the labels for the confusion matrix, then get the matrix itself
    unique_genres = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=unique_genres)

    #Custom plot function
    cm_plot(cm, unique_genres, filename)

    return X_pca, y  #We will return this so we can use it in ROC

#Returns the primary genre in the group of genres
def get_primary_genre(genre):
    if isinstance(genre, list):
        return genre[0]
    #Convert string representation of list into list
    elif isinstance(genre, str) and genre.startswith('['):
        genre_list = eval(genre)
        return genre_list[0]
    else:
        return str(genre)

#Plot the confusion matrix
def cm_plot(cm, classes, filename): 
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels = classes, yticklabels = classes)
    plt.xlabel('Predicted', fontsize=24)
    plt.ylabel('Actual', fontsize=24)
    plt.title('Confusion Matrix', fontsize=30, pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
