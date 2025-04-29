from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MusicRecommender:
    def __init__(self, df, features, pca_df, pca_model, scaler):
        """
        Initializing the recommender system
        
        Input:
        df: Preprocessed and cleaned df
        features: Original audio features used
        pca_df: df with pre-trained PCA features
        pca_model: Pre-trained PCA model
        scaler: Pre-created StandardScaler object
        """
        self.df = df
        self.features = features
        self.pca_df = pca_df
        self.pca_model = pca_model
        self.pca_features = [col for col in pca_df.columns if col.startswith('PC')]
        self.scaler = scaler

    def compute_song_similarity(self, track_id, n_recommendations):
        #Compute cosine similarity matrix for n songs
        song = self.pca_df[self.pca_df['track_id'] == track_id]
        
        if len(song) == 0:
            return "Track not found in the dataset."
        
        #Get song's data for the targeted features and find similarity between all the other songs
        song_pca_features = song[self.pca_features].values
        similarities = cosine_similarity(song_pca_features, self.pca_df[self.pca_features])[0]
        
        #Give them indices
        similarity_scores = list(enumerate(similarities))
        
        #Sort the recommendations excluding the input song itself
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:n_recommendations + 1]
        
        #Get recommendations
        song_indices = [i[0] for i in similarity_scores]
        recommendations = self.pca_df.iloc[song_indices][['track_name', 'artist_name', 'genre']]
        recommendations['similarity_score'] = [i[1] for i in similarity_scores]
        
        return recommendations    
    
    def recommend_by_song(self, track_id, n_recommendations):
        """
        Take the track ID from a user selected song for recommendations
    
        Input:
        track_id: ID of the track the user is interested in
        n_recommendations: Number of recommendations to return (1-5)
        """

        return self.compute_song_similarity(track_id, n_recommendations)
    
    def recommend_by_features(self, feature_values, n_recommendations):
        """
        Recommend songs based on user-specified feature values
        """
        #Create a feature vector from the user inputs
        filled_feature_values = []
        
        for feature in self.features:
            if feature in feature_values:
                filled_feature_values.append(feature_values[feature])
            else:
                #Use median value if not specified
                filled_feature_values.append(self.df[feature].median())
        
        #Reshape and scale
        filled_feature_values = np.array(filled_feature_values).reshape(1, -1)
        scaled_features = self.scaler.transform(filled_feature_values)
        
        #Transform to PCA space and calculate similarity
        pca_features = self.pca_model.transform(scaled_features)
        similarities = cosine_similarity(pca_features, self.pca_df[self.pca_features])[0]
        
        #Add similarity scores and get recommendations
        temp_df = self.pca_df.copy()
        temp_df['similarity'] = similarities
        recommendations = temp_df.sort_values('similarity', ascending=False).head(n_recommendations)
        
        return recommendations[['track_name', 'artist_name', 'genre', 'similarity']]