import pandas as pd
from sklearn.preprocessing import StandardScaler

def combine(path1, path2):
    #Standardizes the column names of both datasets and combines them
    #Combining Datasets --> 0.5 Points
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1 = fix_column_names(df1)
    df2 = fix_column_names(df2)
    
    starting_df = pd.concat([df1, df2], ignore_index=True)
    starting_df = starting_df.drop('Unnamed: 0', axis=1)
    return starting_df

def fix_column_names(df):
    """
    Fixing the:
        Song name
        Artist name
        Track ID
        Genre
    """

    #Map the possible incorrect column names to the one we actually use later
    column_name_map = {
        'name': 'track_name',
        'title': 'track_name',
        'song_name': 'track_name',
        
        'artist': 'artist_name',
        'artists': 'artist_name',
        
        'id': 'track_id',
        
        'track_genre': 'genre'
    }
    
    #Rename those columns
    for old_name, new_name in column_name_map.items():
        if old_name in df.columns:
            df = df.rename(columns = {old_name: new_name})
    
    return df

def fix_genre_names(genre):
    #Some repeat genres are due to capitalization discrepancies
    genre = str(genre).lower().strip()
    
    #Map all the various 'repeat' genre names in the datasets to a standardized name
    genre_map = {
        'r-n-b': 'r&b',
        'rap': 'hip-hop',
        'synth-pop': 'synthpop',
        'sleep': 'ambient',
        'movie': 'soundtrack',
        'show-tunes': 'soundtrack',
        'electro': 'electronic',
        'songwriter': 'singer-songwriter',
        'j-dance': 'j-pop',
        'chill': 'downtempo',
        'latino': 'latin',
        'world-music': 'world'
    }
    
    #Return the correct genre name to be used below
    if genre in genre_map:
        return genre_map[genre]
    
    return genre

def process_genres(df):
    """
    The raw dataset before preprocessing / cleaning has multiple entries of the same song
    with one of their genres per entry. We need to get rid of unnecessary genres and group
    the important genres into one entry.
    """

    #Genres that provide less important info about a track should be removed
    useless_genres = ['chill', 'club', 'goth', 'groove', 'guitar', 'happy', 'piano', 'sad',
                      'brazil', 'british', 'french', 'german', 'spanish', 'swedish', 'Children\'s Music',
                      'children', 'kids', '\'Children\'s Music\'' 
                    ]
    
    #Group repeat tracks
    df['group_cols'] = df['track_name'] + ' by ' + df['artist_name']
    dupe_group = df.groupby('group_cols')
    
    #Process the genres for each separate group of tracks
    def trim_genres(genres):
        
        genres_list = []
        for genre in genres:
            #Skip unwanted genres and fix any wanted genre names to be what we want
            if genre.lower() not in useless_genres:
                fixed_genre = fix_genre_names(genre)
                genres_list.append(fixed_genre)
        
        #Feed the unique genres to create_compound_genres() for further processing
        unique_genres = list(set(genres_list))
        compound_genres = create_compound_genres(unique_genres)

        return compound_genres
    
    #Apply above function to each group of genres to process them
    genre_grouped = dupe_group['genre'].apply(trim_genres).reset_index()
    
    #Take one instance of the duplicated track and merge it with the list of genres
    primary_track_info = df.drop_duplicates(subset='group_cols').drop(columns=['genre'])
    processed_df = primary_track_info.merge(genre_grouped, on='group_cols')
    
    #Any leftover tracks that have no genres we'll remove
    processed_df = processed_df[processed_df['genre'].apply(lambda x: len(x) > 0)]
    
    return processed_df

def create_compound_genres(genres_list):
    """
    Some tracks' genre naming conventions leads to redundancy. For example, a track with genres
    'dance' and 'pop', when 'dance-pop' is a real genre but not represented in the dataset. This will
    hurt our predictive model due to lots of unnecessary noise and failure to seperate the genres
    in a meaningful way. This is why we will create compound genres.

    Some genres are newly created here like 'dance-pop' and some already exist in the dataset like
    'alt-rock' and 'edm'.
    """

    #Define combinations to look for and give a single genre name to represent them
    genre_combinations = {
        ('pop', 'dance'): 'dance-pop',
        ('pop', 'rock'): 'pop-rock',
        ('electronic', 'dance'): 'edm',
        ('indie', 'rock'): 'indie-rock',
        ('indie', 'pop'): 'indie-pop',
        ('alternative', 'rock'): 'alt-rock',
        ('alternative', 'metal'): 'alt-rock',
        ('rap', 'rock'): 'rap-rock',
        ('rap', 'metal'): 'rap-rock',
        ('rock-n-roll', 'rock'): 'rock-n-roll',
        ('rockabilly', 'rock'): 'rockabilly',
        ('anime', 'rock'): 'anime',
        ('grunge', 'rock'): 'grunge'
    }
    
    genres_set = set(genres_list)
    result_genres = genres_set.copy()

    #Track which combinations we've done already
    applied_combos = []
    
    #Checks all the combinations and adds compound genres for target combinations
    for (genre1, genre2), compound_genre in genre_combinations.items():
        if genre1 in genres_set and genre2 in genres_set:
            result_genres.add(compound_genre)
            applied_combos.append(((genre1, genre2), compound_genre))
    
    #After all compounds are added, start removing the originals that we no longer want
    for (genre1, genre2), compound_genre in applied_combos:
        if genre1 in genres_set and genre2 in genres_set:
            #Make sure one of the original genres is not one we want to keep
            if compound_genre == genre1 and genre2 in result_genres:
                result_genres.remove(genre2)
            elif compound_genre == genre2 and genre1 in result_genres:
                result_genres.remove(genre1)
            else:
            #If both original genres need to be removed
                if genre1 in result_genres:
                    result_genres.remove(genre1)
                if genre2 in result_genres:
                    result_genres.remove(genre2)
    
    
    return list(result_genres)

def has_unwanted_genre(genres):
#Trim tracks that contain a genre we are not interested in from the dataset
   
    unwanted_genres = ['cantopop', 'mandopop', 'indian', 'party', 'iranian', 'malay',
                       'romance', 'turkish', 'soundtrack'   
    ]

    #Because of the genre processing some tracks may have genres as a list
    if isinstance(genres, list):
        #If track has any of our unwanted genres return true
        for genre in genres:
            if genre in unwanted_genres:
                return True
        
        return False
    else:
        #If a track has one genre
        return genre in unwanted_genres

def clean(df):
    #Data cleaning
    df = df.dropna(subset=['track_name', 'artist_name', 'genre'])
    """
    I was having trouble getting all the genres related to children's music out and it was clogging
    the dataset pretty bad because of useless genre assignments, but I am out of time so I just
    hard coded it in
    """
    df = df[~df['genre'].str.lower().str.contains('children|kids')]

    #Reduce noise by trimming unneeded genres or condensing them into more useful ones
    df = process_genres(df)

    """As of now I have not completely cleaned up an issue of genres existing as strings, lists, or 
    string representations of lists at different points. So for now the handling is a bit messy.
    
    Here I am making sure the compound genre lists created by calling process_genres() are actual lists.
    Then I prune any songs that contain an unwanted genre with has_unwanted_genre().
    """
    df['genre'] = df['genre'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df = df[~df['genre'].apply(has_unwanted_genre)]
    
    df = df.drop_duplicates(subset=['track_id'])
    df = df.drop_duplicates(subset=['track_name', 'artist_name'])
    
    #If a feature is numerical, fill NaN with median
    num_features = ['acousticness', 'danceability', 'duration_ms', 'energy', 
            'instrumentalness', 'liveness', 'loudness', 'tempo', 'valence']
    
    for f in num_features:
        df[f] = df[f].fillna(df[f].median())
   
    #Changing ms into seconds now will make it easy in Streamlit
    df['duration_sec'] = df['duration_ms'] / 1000
    
    return df

def scale(df, features):
    scaler = StandardScaler()
    scaled_df = df.copy()
    scaled_df[features] = scaler.fit_transform(df[features])
    #We will use the scaler object we made later
    
    return scaled_df, scaler