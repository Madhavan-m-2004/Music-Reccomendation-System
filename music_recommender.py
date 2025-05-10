import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
def load_data(filepath):
    return pd.read_csv(filepath)

# Create a user-song matrix
def create_user_song_matrix(df):
    return pd.crosstab(df['user_id'], df['song_name'])

# Collaborative Filtering
def get_collab_recommendations(user_id, user_song_matrix, top_n=3):
    similarity = cosine_similarity(user_song_matrix)
    sim_df = pd.DataFrame(similarity, index=user_song_matrix.index, columns=user_song_matrix.index)
    
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:]  # exclude self
    top_users = similar_users.head(top_n).index
    
    songs = user_song_matrix.loc[top_users].sum().sort_values(ascending=False)
    already_listened = user_song_matrix.loc[user_id]
    recommended = songs[already_listened == 0].head(5).index.tolist()
    
    return recommended

# Content-Based Filtering
def get_content_recommendations(song_name, df):
    song_info = df[df['song_name'] == song_name][['artist', 'genre']].iloc[0]
    df['combined'] = df['artist'] + " " + df['genre']
    
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(df['combined'])
    
    similarity = cosine_similarity(count_matrix)
    song_index = df[df['song_name'] == song_name].index[0]
    similar_indices = similarity[song_index].argsort()[::-1][1:6]
    
    similar_songs = df.iloc[similar_indices]['song_name'].unique().tolist()
    return similar_songs
