import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import seaborn as sns

import streamlit as st

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import manifold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import plotly.graph_objects as go

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

sns.set_style("darkgrid")
sns_p = sns.color_palette('bright')

# ==============================================

from PIL import Image
import requests
from io import BytesIO

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/wf4.png")
img = Image.open(BytesIO(response.content))#.resize((900, 100))



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

st.markdown('This app provides interactive visual reports, NLP (natural language processing) analysis and statistical summaries based on the national anthem lyrics of more than 100 countries. Start by selecting a country from left panel.')
'Data scraped from ', 'http://www.nationalanthems.info'
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

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/neut.png")
    neut = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(neut, use_column_width = False)

elif country_sentiment == "POSITIVE":

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/pos.png")
    pos = Image.open(BytesIO(response.content)).resize((200, 200))
    st.image(pos, use_column_width = False)

else:

    response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/neg.png")
    neg = Image.open(BytesIO(response.content)).resize((200, 200))
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

"# Most important words in national anthem lyrics of ", country
i_important_words = st.slider('Select number of prominant words to show', 3, 12, 3)
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

st.markdown('---------------------------------------------------------------------------')
"# Clustering of countries based on their national anthem lyrics"
st.markdown('### t-Distributed Stochastic Neighbor Embedding (t-SNE)')
st.markdown('t-SNE is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two (or 3D) map.')

#call from saved
#tsne = manifold.TSNE(n_components = 2, init = 'pca', random_state = 420)
#Y = tsne.fit_transform(X.toarray())
#data = pd.DataFrame(Y, columns=['Y0', 'Y1'])
#data['Country'] = countries

kk = st.slider(f'Select a number for clustering', 5, 55, 10, 1)

st.markdown('Interactive map of clustering of countries based on their national anthem lyrics.')

data = pd.read_csv('data_label.csv')
label_kk = 'Label' + str(kk)


#KMI = []

#for n in [kk]:#range(5, 120, 5):

#    km = KMeans(n_clusters = n)
#    km.fit(data[['Y0', 'Y1']])
#    KMI.append(km.inertia_)

#data['Label'] = km.labels_

fig = go.Figure(data=go.Scatter(x = data['Y0'],
                                y = data['Y1'],
                                mode = 'markers',
                                marker = dict(color = data[label_kk], colorscale='Rainbow'),
                                text = data['Country'], opacity = 0.92, marker_symbol = data['Label']))

fig.update_traces(marker = dict(size = 13,
                                line = dict(width = 2.0,
                                          color = 'black')))

fig.update_layout(margin=dict(l=80, r=80, t=100, b=80), height = 700, width = 700)

st.plotly_chart(fig)


st.markdown('---------------------------------------------------------------------------')
st.markdown('## Cosine of Similarity Matrix')

"Cosine of similarity is a measure of similarity between two non-zero vectors () of an inner product space."

show_cos = st.checkbox('Show cosine of similarity matrix')

if show_cos:

    fig = plt.figure(figsize = (5, 3))
    plt.imshow(np.array(COS_sim_m), cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.grid('off')
    st.pyplot(fig)

# ======================================================================================================
"# Word-count per national anthem"
i_top_word_count = st.slider('Select number of countries with highest word-count to show', 7, 32, 7)

na_df_sorted = na_df.sort_values(by = 'Word Count', ascending = False)


fig = plt.figure(figsize = (5, 3))

sns.barplot(x = "Word Count", y ="Country", data = na_df_sorted[:i_top_word_count], palette = sns_p, edgecolor = 'k')
plt.margins(0.1)
#plt.tight_layout()
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
