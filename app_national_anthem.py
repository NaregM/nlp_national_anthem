import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import seaborn as sns

import streamlit as st

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

sns.set_style("darkgrid")
sns_p = sns.color_palette('Paired')

# ==============================================

from PIL import Image
import requests
from io import BytesIO

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/333.png")
img = Image.open(BytesIO(response.content))

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/neut.png")
neut = Image.open(BytesIO(response.content)).resize((200, 200))

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/pos.png")
pos = Image.open(BytesIO(response.content)).resize((200, 200))

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/neg.png")
neg = Image.open(BytesIO(response.content)).resize((200, 200))


#image = Image.open('/home/nareg/Downloads/cmb-1.jpg')
st.image(img, use_column_width = True)
# ===============================================

na_df = pd.read_csv('national_anthems.csv', index_col = 0)

countries = na_df.index.tolist()
nanthems = na_df['National Anthem'].tolist()

country_anthem = defaultdict(list)

for i, country_ in enumerate(countries):
    
    country_anthem[country_].append(nanthems[i])

country_anthem_all = []
for key in country_anthem.keys():
    
    country_anthem_all.append(' '.join(country_anthem[key]).encode('utf8'))

vectorizer = TfidfVectorizer(min_df = 2,
                             ngram_range = (1, 3),
                             strip_accents = 'unicode',
                             stop_words = 'english',
                             max_features = 5000)

X = vectorizer.fit_transform(country_anthem_all)

vocab = vectorizer.get_feature_names()


st.markdown('## **_National Anthem_: An NLP and Statistical  Analysis**')

st.markdown('This app provides interactive visual reports, NLP analysis and statistical summaries based on the national anthem of more than 300 countries. Start by selecting a country from left panel.')
st.markdown('-------------------------------------------------------------------------------')



df0 = pd.DataFrame({
                   'Choose a Country': countries
                   })


country = st.sidebar.selectbox(
    'Select a country: ',
    countries)


country_lyrics = na_df[na_df.index == country]['National Anthem'].values[0]

country_sentiment = na_df[na_df.index == country]['sentiment'].values[0]

country_id = np.where(na_df.index == country)[0][0]


"## __The lyrics of national anthem of__ ", country, " : ", "\n" 
st.write(country_lyrics)


"## __Sentiment analysis result for__ ", country, 
"Possible outcomes: Negative, Neutral, Positive"
"## -->> ", country_sentiment

if country_sentiment == "NEUTRAL":
    st.image(neut, use_column_width = False)

elif country_sentiment == "POSITIVE":
    st.image(pos, use_column_width = False)
else:
    st.image(neg, use_column_width = False)




show_map = st.checkbox('Show distribution of sentiment for all countries')

if show_map:


    fig = plt.figure(figsize = (5, 2))

    plt.bar(np.unique(na_df['sentiment'].values, return_counts = True)[0],
            np.unique(na_df['sentiment'].values, return_counts = True)[1], color = 'dodgerblue')

    st.pyplot(fig)


if st.checkbox('Show countries with Negative outcome'):

    na_df['Country'][na_df['sentiment'] == 'NEGATIVE'].index

if st.checkbox('Show countries with Positive outcome'):

    na_df['Country'][na_df['sentiment'] == 'POSITIVE'].index



st.markdown('---------------------------------------------------------------------------')



st.markdown('---------------------------------------------------------------------------')


cond_non_zero = np.where(pd.DataFrame(data = X.toarray(), columns=vocab).iloc[country_id, :] != 0)

"# Most important words in national anthem lyrics of ", country, " : "
i_important_words = st.slider('Number of prominant words: ', 3, 12, 3)
df_tfidf = pd.DataFrame(np.array(vocab)[cond_non_zero][:i_important_words], columns = ['Prominant Words'])

st.write(df_tfidf.astype('object'))


# ================================================================================
def max_n(V, n):
    
    """
    Returns the n largest elements from a numpy array.
    """
    argmax = []
    
    VV = np.copy(V)
    
    for i in range(n):
        
        argmax.append(np.argmax(VV))
        VV[np.argmax(VV)] = -9999
        
    return V[argmax]


def argmax_n(V, n):
    
    """
    Returns the n largest indices from a numpy array.
    """
    argmax = []
    
    VV = np.copy(V)
    
    for i in range(n):
        
        argmax.append(np.argmax(VV))
        VV[np.argmax(VV)] = -9999
        
    return argmax

"# Countries with most similar lyrics to ", country, " "

COS_sim_m = (np.inner(X.toarray(), (X.toarray()))/(np.linalg.norm(X.toarray()) * np.linalg.norm(np.transpose(X.toarray()), axis = 0)))

i_top_match = st.slider(f'Select number of countries with most similarity', 4, 12, 4)

st.dataframe(np.array(countries)[argmax_n(COS_sim_m[country_id, :], i_top_match+1)][1:])

if True:
    
    st.markdown('---------------------------------------------------------------------------')
    st.markdown('## Cosine of Similarity Matrix')

    fig = plt.figure(figsize = (5, 3))
    plt.imshow(np.array(COS_sim_m), cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.grid('off')
    st.pyplot(fig)

# ======================================================================================================
"# Word-count per national anthem"
i_top_word_count = st.slider('Select number of countries to show', 3, 12, 3)

na_df_sorted = na_df.sort_values(by = 'Word Count', ascending = False)


fig = plt.figure(figsize = (5, 3))

sns.barplot(x = "Word Count", y ="Country", data = na_df_sorted[:i_top_word_count], palette = sns_p, edgecolor = 'k')
plt.margins(0.1)
st.pyplot(fig)


if False:
    st.markdown('---------------------------------------------------------------------------')
    #df


st.markdown('-------------------------------------------------------------------------------')

st.markdown("""
                Made by [Nareg Mirzatuny](https://github.com/NaregM)

Source code: [GitHub](
                https://github.com/NaregM/nlp_national_anthem)

""")
st.markdown('-------------------------------------------------------------------------------')

