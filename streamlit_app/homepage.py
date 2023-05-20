import streamlit as st
import base64
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Animendation', layout='wide')

st.title('Welcome to Animendation!')
file_ = open('gifs/oshi-no-ko-ruby.gif', "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
st.divider()

@st.cache_resource(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df
cleaned_df = load_data('data/cleaned_anime_data.csv')
labels = list(cleaned_df['title'])
labels.insert(0, '')
user_input = st.selectbox(label='Enter an anime', options=labels)
generate_rec = st.button('Generate Recommendations')

if generate_rec:
    @st.cache_resource(show_spinner=False)
    def create_cos_sim(string_column):
        tf_vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.7, min_df=3)
        tfidf_matrix = tf_vec.fit_transform(string_column)
        tfidf_array = tfidf_matrix.toarray()
        return cosine_similarity(tfidf_array, tfidf_array)

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
        
        # filter out animes that have a mean (rating) lower than 7
        filtered_results = results[results['mean']>=7]
        # check if results contain the words in the title, if yes, remove and add new recommendation until no sequels exists in the top 20
        title_word_list = [x for x in title.split() if len(x)>5]
        for i in title_word_list:
            filtered_results = filtered_results[filtered_results['title'].str.contains(i)==False]
        # filtered_results = results[results['title'].str.contains(title)==False]

        diff = results.shape[0] - filtered_results.shape[0]
        
        if diff == 0:
            return filtered_results
        else:
            add_recommendations = pd.DataFrame(columns=['title', 'alternative_title(s)', 'synopsis', 'genres', 'studios', 'mean', 'year', 'status', 'medium_picture_url'])
            while diff > 0:
                new_idx = sim_scores[last:last+diff]
                last = last + diff
                new_movie_idx = [i[0] for i in new_idx]
                new_recommendations = cleaned_df[['title', 'alternative_title(s)', 'synopsis', 'genres', 'studios', 'mean', 'year', 'status', 'medium_picture_url']].iloc[new_movie_idx]
                filtered_new_recommendations = new_recommendations[new_recommendations['mean']>=7]
                for i in title_word_list:
                    filtered_new_recommendations = filtered_new_recommendations[filtered_new_recommendations['title'].str.contains(i)==False]
                # filtered_new_recommendations = filtered_new_recommendations[filtered_new_recommendations["title"].str.contains(title)==False]

                add_recommendations = pd.concat([add_recommendations, filtered_new_recommendations])    
                diff = new_recommendations.shape[0] - filtered_new_recommendations.shape[0]

            new_results = pd.concat([filtered_results, add_recommendations])
            return new_results

    with st.spinner('Calculating and generating recommendations...'):
        cos_sim = create_cos_sim(cleaned_df['cleaned_string'].astype('str'))
        indices = pd.Series(cleaned_df.index, index=cleaned_df['title'])

    animes = get_recommendations(user_input, cos_sim, indices).sort_values(by=['year', 'mean'], ascending=False).reset_index(drop=True)
    animes.index += 1

    st.divider()

    for x in range(1,21):
        st.write(f'Recommendation {x}: ')
        with st.expander(animes['title'].loc[x], expanded=True):
            col1, col2 = st.columns([1,4])
            with col1:
                st.image(animes['medium_picture_url'].loc[x])
            with col2:
                st.header(animes['title'].loc[x])
                st.write('Alternative title(s):', str(animes['alternative_title(s)'].loc[x]))
                st.write('Genre(s):', animes['genres'].loc[x])
                st.write('Studio(s):', animes['studios'].loc[x])
                st.write('Year released:', str(animes['year'].loc[x]))
                st.write('Status:', animes['status'].loc[x])
                st.write('Synopsis:', animes['synopsis'].loc[x])




