import pickle
import pandas as pd

"""
This pipeline is responsible for actually executing all of the data preprocessing and 
model training. All created models (scaler, PCA, KNN) are saved for future use as pickle
objects.


Methods from dp: combine, clean, scale
Methods from fe: do_pca
Methods from gc: do_knn
Methods from val: cross_validation
"""

import src.data_processing as dp
import src.feature_engineering as fe
import src.genre_classification as gc
import src.validation as val

def main():
    print("Starting data processing and model training.")
    
    #Data preprocessing
    print("Loading datasets...")
    try:
        #Combine the two datasets
        combined_df = dp.combine("data/raw/spotify_dataset1.csv", "data/raw/spotify_dataset2.csv")
        print(f"Combined dataset shape: {combined_df.shape}")

        #Save combined data
        combined_path = 'data/processed/combined_data.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"Processed data saved to {combined_path}")
        
        #Preprocess, clean data
        print("\nPreprocessing and cleaning...")
        processed_df = dp.clean(combined_df)
        print(f"Cleaned dataset shape: {processed_df.shape}")
        
        #Save processed data
        processed_path = 'data/processed/processed_data.csv'
        processed_df.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")
        
        #Define what features we will use for training
        features = ['acousticness', 'danceability', 'duration_sec', 'energy', 
                   'instrumentalness', 'liveness', 'loudness', 'tempo', 'valence']
        
        #Scale features
        print("Scaling features...")
        scaled_df, scaler = dp.scale(processed_df, features)
        print(f"Scaled dataset shape: {scaled_df.shape}")
        
        #Save scaler so it can be reused each time we open the Streamlit app
        scaler_filename = 'models/scaler.pkl'
        pickle.dump(scaler, open(scaler_filename, 'wb'))
        print(f"Scaler saved to {scaler_filename}")
        
        #Feature engineering
        print("\nStarting feature engineering...")
        print("Performing PCA...")
        pca_df, pca_model = fe.do_pca(scaled_df, features, 8)
        print(f"PCA dataset shape: {pca_df.shape}")
        
        #Save PCA model for the same reason
        pca_filename = 'models/pca_model.pkl'
        pickle.dump(pca_model, open(pca_filename, 'wb'))
        print(f"PCA model saved to {pca_filename}")

        #Save PCA data
        pca_path = 'data/processed/pca_data.csv'
        pca_df.to_csv(pca_path, index=False)
        print(f"Processed data saved to {pca_path}")
        
        #Genre classification
        print("\nTraining KNN classifier...")
        knn_features = [f'PC{i+1}' for i in range(8)]
        knn_model, X_train, X_test, y_train, y_test, y_pred, cm = gc.multi_genre_knn(
            pca_df, knn_features, target='genre', k=12
        )

        #Save KNN model
        knn_filename = 'models/knn_model.pkl'
        pickle.dump(knn_model, open(knn_filename, 'wb'))
        print(f"KNN model saved to {knn_filename}")
        
        #Validation
        print("\nPerforming cross-validation...")
        cv_scores = val.cross_validation(knn_model, X_train, y_train, 5)
        
        print("\nPipeline completed successfully!")
        print("\nRun the recommender system in Streamlit with:")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        
if __name__ == "__main__":
    main()