import streamlit as st
import pandas as pd
import pickle
from src.recommender import MusicRecommender

#Dictionary explaining audio features - pulled from Spotify API
features_dictionary = {
    'acousticness': 'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.',
    'danceability': 'Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.',
    'energy': 'A measure from 0.0 to 1.0 representing intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.',
    'instrumentalness': 'Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.',
    'liveness': 'Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.',
    'loudness': 'The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track. Values typically range between -60 and 0 db.',
    'tempo': 'The overall estimated tempo of a track in beats per minute (BPM).',
    'valence': 'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).',
    'duration_sec': 'The duration of the track in seconds.'
}

def load_model(model_path):
    #Load saved model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model

def load_data(data_path):
    #Load processed data
    df = pd.read_csv(data_path, low_memory=False)
    #Loading the CSV back in causes the genre lists to become strings, so we de-nest it
    df['genre'] = df['genre'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)

    return df

def main():
    #Main app description
    st.title("Music Recommendation System")
    st.write("Get personalized song recommendations based on user input")
    
    try:
        #Load models
        df = load_data('data/processed/processed_data.csv')
        pca_df = load_data('data/processed/pca_data.csv')
        knn_model = load_model('models/knn_model.pkl')
        pca_model = load_model('models/pca_model.pkl')
        scaler = load_model('models/scaler.pkl')
        
        features = ['acousticness', 'danceability', 'duration_sec', 'energy', 
                   'instrumentalness', 'liveness', 'loudness', 'tempo', 'valence']
        
        #Initialize recommender
        recommender = MusicRecommender(df, features, pca_df, pca_model, scaler)
        
        #Each recommendation method will have a tab, as well as the feature dictionary
        tab1, tab2, tab3 = st.tabs(["Recommend by Song", "Recommend by Features", "Feature Descriptions"])
        

        #First tab for recommendation by user selected song
        with tab1:
            st.header("Find Similar Songs")
    
            #Create a search bar for songs and filter by song title or artist name
            searching = st.text_input("Search for a song:", "")
    
            #Narrow to all songs that match user input (autofill))
            if searching:
                filtered_songs = df[
                    (df['track_name'].str.contains(searching, case=False)) | 
                    (df['artist_name'].str.contains(searching, case=False))
                    ]
                #We don't want to display thousands of options at once, limit to 30
                filtered_songs = filtered_songs.head(30)
        
                if len(filtered_songs) > 0:
                    #Show the user which songs match their inputted text
                    song_options = [f"{row.track_name} by {row.artist_name}" for row in filtered_songs.itertuples()]
                    #Prompt user to pick one of the displayed options
                    selected_option = st.selectbox("Select a song from search results:", song_options)
            
                    #Get the index of that song and its track ID
                    selected_index = song_options.index(selected_option)
                    selected_track_id = filtered_songs.iloc[selected_index]['track_id']
            
                    #Display song details
                    st.write(f"Selected: **{filtered_songs.iloc[selected_index]['track_name']}** by *{filtered_songs.iloc[selected_index]['artist_name']}*")
                    
                    #User selects how many recommendations they want
                    n_recommendations = st.slider("Number of recommendations:", 1, 5, 3)
            
                    #User chooses to get recommendations with a button
                    if st.button("Get Recommendations", key = "song_rec_button"):
                        recommendations = recommender.recommend_by_song(selected_track_id, n_recommendations)
                
                        #Display recommendations
                        st.subheader("Recommended Songs:")
                        for i, row in enumerate(recommendations.itertuples()):
                            if isinstance(row.genre, list):
                                genres = ", ".join(row.genre)
                            else:
                                genres = str(row.genre)
    
                            st.write(f"{i+1}. **{row.track_name}** by *{row.artist_name}*")
                            st.write(f"Genre: {genres}")
                            st.write(f"   Similarity: {row.similarity_score:.2f}")
                            st.write("---")
                
                        #Used the pretrained models to scale and put the song in PCA space
                        selected_song_features = filtered_songs.iloc[selected_index][features].values.reshape(1, -1)
                        selected_song_scaled = scaler.transform(selected_song_features)
                        selected_song_pca = pca_model.transform(selected_song_scaled)

                        #Predict the song's genre and compare it to its actual genre
                        predicted_genre = knn_model.predict(selected_song_pca)[0]

                        actual_genre = filtered_songs.iloc[selected_index]['genre']
                        if isinstance(actual_genre, list):
                            actual_str = ", ".join(actual_genre)
                        else:
                            actual_str = str(actual_genre)
                
                        #Display prediction info
                        st.subheader("Genre Prediction:")
                        st.write(f"Actual Genre: **{actual_str}**")
                        st.write(f"Predicted Genre: **{predicted_genre}**")

                else:
                    st.info("No matching songs found in the database.")
            else:
                st.info("Enter a song or artist name to search")

        #Second tab for recommendation by user specified song features
        with tab2:
            #Description of sliders
            st.header("Find Songs by Features")
            st.subheader("Adjust the sliders to set your preferred song characteristics:")
            st.write("Hover over each feature name to see what it is")
            
            feature_values = {}
            
            #Define ranges for each feature
            feature_ranges = {
                'acousticness': (0.0, 1.0, df['acousticness'].median()),
                'danceability': (0.0, 1.0, df['danceability'].median()),
                'energy': (0.0, 1.0, df['energy'].median()),
                'instrumentalness': (0.0, 1.0, df['instrumentalness'].median()),
                'liveness': (0.0, 1.0, df['liveness'].median()),
                'loudness': (-60.0, 0.0, df['loudness'].median()),
                'tempo': (50.0, 200.0, df['tempo'].median()),
                'valence': (0.0, 1.0, df['valence'].median())
            }
            
            #Create a slider for each feature
            for feature, (min_val, max_val, default_val) in feature_ranges.items():
                feature_values[feature] = st.slider(
                    f"{feature.capitalize()}", 
                    min_val, 
                    max_val,
                    float(default_val),
                    key = f"{feature}_slider",
                    help = features_dictionary[feature]
                )
            
            #User inputs duration in 'minutes:seconds' and it's converted to seconds
            min_input = st.text_input('Duration in (minutes:seconds)', '4:00', 
                                       help = features_dictionary['duration_sec'])
            try:
                #Reformat the input back to seconds
                minutes, seconds = min_input.split(':')
                feature_values['duration_sec'] = (int(minutes) * 60) + int(seconds)

            except:
                #If the user inputs an invalid duration, just use an arbitary duration (chosen as 4 minutes)
                feature_values['duration_sec'] = 240 
            
            #Number of recommendations
            n_recommendations = st.slider("Number of recommendations:", 1, 5, 3, key = "feature_rec_slider")
            
            #User chooses to get recommendations with a button
            if st.button("Get Recommendations", key = "feature_rec_button"):
                recommendations = recommender.recommend_by_features(feature_values, n_recommendations)
                
                #Display recommendations
                st.subheader("Recommended Songs:")
                for i, row in enumerate(recommendations.itertuples()):
                    if isinstance(row.genre, list):
                        genres = ", ".join(row.genre)
                    else:
                        genres = str(row.genre)
    
                    st.write(f"{i+1}. **{row.track_name}** by *{row.artist_name}*")
                    st.write(f"Genre: {genres}")
                    st.write(f"   Similarity: {row.similarity:.2f}")
                    st.write("---")
        
        #Information tab explaining what the features are, and what the recommender system does
        with tab3:
            st.header("Explanation of Recommendations by Musical Features")
            st.write("These are the audio features used by Spotify to characterize their songs:")
            
            #Create df and display dictionary
            features_df = pd.DataFrame({
                'Feature': features_dictionary.keys(),
                'Description': features_dictionary.values()
            })
            st.table(features_df)
            
            #Add some additional context
            st.subheader("How Recommendations are Made")
            st.write("""
                This recommender system offers a set of sliders corresponding to each of the above list of musical features.
                A user can specify a specific preferred value for each musical feature. If a slider is not moved by the user,
                for example, 'acousticness,' the recommender system assumes the user prefers the median value of acousticness for 
                all songs in the database. Recommendations are made based on all features and their user-specified values. So, for
                example, if the user sets a high value for 'acousticness' and a low value for 'energy,' they will be recommended
                songs that have both a high 'acousticness' score and a low 'energy' score, rather than any song that has either a
                high 'acousticness' score or a low 'energy' score.
            """)
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Check that the proper models have been trained and exist in the correct path.")

if __name__ == "__main__":
    main()