import pandas as pd
from sklearn.decomposition import PCA

def do_pca(df, features, n):
    #PCA (0.5 Point)
    #Unfinished: How to find optimal number of components (besides trying different values)

    pca = PCA(n_components = n)
    components = pca.fit_transform(df[features])
    
    #Pair the components back with the song info
    c = [f'PC{i+1}' for i in range(n)]
    pca_df = pd.DataFrame(data = components, columns = c)
    
    for c in ['track_id', 'track_name', 'artist_name', 'genre']:
        pca_df[c] = df[c].values
    
    #How did our PCA do
    var = pca.explained_variance_ratio_
    total_var = sum(var)
    print(f'Explained variance ratios: {var}')
    print(f'Total explained variance: {total_var}')
    
    return pca_df, pca