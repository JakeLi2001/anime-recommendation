import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Animendations', layout='wide')

st.title('Enter an anime title and get recommendations based on plot similarity!')
cleaned_df = pd.read_csv('data/cleaned_anime_data.csv')
labels = cleaned_df['title']
user_input = st.selectbox(label='Enter an anime', options=labels)

tf_vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=0.1, use_idf=True)
tfidf_matrix = tf_vec.fit_transform(cleaned_df['cleaned_string'])
tfidf_array = tfidf_matrix.toarray()
cos_sim = cosine_similarity(tfidf_array, tfidf_array)
indices = pd.Series(cleaned_df.index, index=cleaned_df['title'])

def get_recommendations(title, cosine_sim, indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    first = 1
    last = 21
    top_20 = sim_scores[first:last]
    movie_indices = [i[0] for i in top_20]
    
    # return the top 20 most similar animes that doesn't contain the title
    results = cleaned_df[['title', 'alternative_title(s)', 'synopsis', 'genres', 'studios', 'mean', 'year', 'status', 'medium_picture_url']].iloc[movie_indices]
    
    # check if results contain the sequels, if yes, remove them and add new recommendation until no sequels exists in the top 20 
    filtered_results = results[results['title'].str.contains(title)==False]
    diff = results.shape[0] - filtered_results.shape[0]
    
    if diff == 0:
        return results
    else:
        add_recommendations = pd.DataFrame(columns=['title', 'alternative_title(s)', 'synopsis', 'genres', 'studios', 'mean', 'year', 'status', 'medium_picture_url'])
        while diff > 0:
            new_idx = sim_scores[last:last+diff]
            last = last + diff
            new_movie_idx = [i[0] for i in new_idx]
            new_recommendations = cleaned_df[['title', 'alternative_title(s)', 'synopsis', 'genres', 'studios', 'mean', 'year', 'status', 'medium_picture_url']].iloc[new_movie_idx]
            filtered_new_recommendations = new_recommendations[new_recommendations["title"].str.contains(title)==False]
            add_recommendations = pd.concat([add_recommendations, filtered_new_recommendations])    
            diff = new_recommendations.shape[0] - filtered_new_recommendations.shape[0]

        new_results = pd.concat([filtered_results, add_recommendations])
        return new_results

animes = get_recommendations(user_input, cos_sim, indices)
print(animes['medium_picture_url'])

st.dataframe(animes)



# st.image(animes['medium_picture_url'][0])
# with st.expander(animes['title'][0]):











