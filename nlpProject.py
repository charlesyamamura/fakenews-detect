#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
from spacy import displacy
from spacy import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel, TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


# set plot options
plt.rcParams['figure.figsize'] = (9, 6)
default_plot_colour = "#00bfbf"


# In[ ]:


data = pd.read_csv("fakeNews.csv")


# In[ ]:


data.info()


# In[ ]:


#data.head(10)


# In[ ]:


data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_colour)
plt.title('Count of Article Classifier')
plt.show()


# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


fake = data[data["fake_or_factual"] == "Fake News"]
fact = data[data["fake_or_factual"] == "Factual News"]


# In[ ]:


fakedocs = list(nlp.pipe(fake['text']))
factdocs = list(nlp.pipe(fact['text']))


# In[ ]:


def extractTokenTags(doc: spacy.tokens.doc.Doc):
    return [(i.text, i.ent_type_, i.pos_) for i in doc]


# In[ ]:


faketags = []
columns = ['token', 'nerTag', 'posTag']


# In[ ]:


for ix, doc in enumerate(fakedocs):
    tags = extractTokenTags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    faketags.append(tags)


# In[ ]:


faketags = pd.concat(faketags)


# In[ ]:


facttags = []
columns = ['token', 'nerTag', 'posTag']


# In[ ]:


for ix, doc in enumerate(factdocs):
    tags = extractTokenTags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    facttags.append(tags)


# In[ ]:


facttags = pd.concat(facttags)


# In[ ]:


facttags.head()


# In[ ]:


# token frequency count (fake)
posCountsFake = faketags.groupby(['token','posTag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
posCountsFake.head(10)


# In[ ]:


# token frequency count (fact)
posCountsFact = facttags.groupby(['token','posTag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
posCountsFact.head(10)


# In[ ]:


# frequencies of pos tags
posCountsFake.groupby(['posTag'])['token'].count().sort_values(ascending=False).head(10)


# In[ ]:


posCountsFact.groupby(['posTag'])['token'].count().sort_values(ascending=False).head(10)


# In[ ]:


posCountsFake[posCountsFake.posTag == "NOUN"][:15]


# In[ ]:


posCountsFact[posCountsFact.posTag == "NOUN"][:15]


# In[ ]:


# top entities in fake news
topFake = faketags[faketags['nerTag'] != ""] \
                    .groupby(['token','nerTag']).size().reset_index(name='counts') \
                    .sort_values(by='counts', ascending=False)


# In[ ]:


topFact = facttags[facttags['nerTag'] != ""] \
                    .groupby(['token','nerTag']).size().reset_index(name='counts') \
                    .sort_values(by='counts', ascending=False)


# In[ ]:


# create custom palette to ensure plots are consistent
nerPalette = {
    'ORG': sns.color_palette("Set2").as_hex()[0],
    'GPE': sns.color_palette("Set2").as_hex()[1],
    'NORP': sns.color_palette("Set2").as_hex()[2],
    'PERSON': sns.color_palette("Set2").as_hex()[3],
    'DATE': sns.color_palette("Set2").as_hex()[4],
    'CARDINAL': sns.color_palette("Set2").as_hex()[5],
    'PERCENT': sns.color_palette("Set2").as_hex()[6]
}


# In[ ]:


sns.barplot(
    x = 'counts',
    y = 'token',
    hue = 'nerTag',
    palette = nerPalette,
    data = topFake[0:10],
    orient = 'h',
    dodge=False
) \
.set(title='Most Common Entities in Fake News')


# In[ ]:


sns.barplot(
    x = 'counts',
    y = 'token',
    hue = 'nerTag',
    palette = nerPalette,
    data = topFact[0:10],
    orient = 'h',
    dodge=False
) \
.set(title='Most Common Entities in Factual News')


# In[ ]:


# a lot of the factual news has a location tag at the beginning of the article, let's use regex to remove this
data['text_clean'] = data['text'].apply(lambda x: re.sub(r"^[^-]*-\s*", "", x))


# In[ ]:


data['text_clean'] = data['text_clean'].str.lower()


# In[ ]:


data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r"[^\w\s]", "", x))


# In[ ]:


data.head()


# In[ ]:


sws = stopwords.words('english')
print(sws)


# In[ ]:


data['text_clean'] = data['text_clean'].apply(lambda x: " ".join(word for word in x.split() if word not in sws))


# In[ ]:


data.head()


# In[ ]:


data['text_clean'] = data['text_clean'].apply(lambda x: word_tokenize(x))


# In[ ]:


lemmatizer = WordNetLemmatizer()
data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token)\
                                             for token in tokens])


# In[ ]:


tokens = sum(data['text_clean'], [])
unigrams = (pd.Series(nltk.ngrams(tokens, 1)).value_counts()).reset_index()[:10]
print(unigrams)


# In[ ]:


unigrams['token'] = unigrams['index'].apply(lambda x: x[0]) # extract the token from the tuple so we can plot it

sns.barplot(x = "count", 
            y = "token", 
            data=unigrams,
            orient = 'h',
            palette=[default_plot_colour],
            hue = "token", legend = False)\
.set(title='Most Common Unigrams After Preprocessing')


# In[ ]:


bigrams = (pd.Series(nltk.ngrams(tokens, 2)).value_counts()).reset_index()[:10]
print(bigrams)


# In[ ]:


bigrams['token'] = bigrams['index'].apply(lambda x: x[0]) # extract the token from the tuple so we can plot it

sns.barplot(x = "count", 
            y = "token", 
            data=bigrams,
            orient = 'h',
            palette=[default_plot_colour],
            hue = "token", legend = False)\
.set(title='Most Common Bigrams After Preprocessing')


# In[ ]:


vader = SentimentIntensityAnalyzer()


# In[ ]:


data['sent_score'] = data["text"].apply(lambda x: vader.polarity_scores(x)['compound'])


# In[ ]:


data.head(15)


# In[ ]:


# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

data['sentiment'] = pd.cut(data['sent_score'], bins, labels=names)


# In[ ]:


data['sentiment'].value_counts().plot.bar(color=default_plot_colour)


# In[ ]:


sns.countplot(
    x = 'fake_or_factual',
    hue = 'sentiment',
    palette = sns.color_palette("hls"),
    data = data
) \
.set(title='Sentiment by News Type')


# In[ ]:


fakeNews = data[data["fake_or_factual"] == "Fake News"]["text_clean"].reset_index(drop=True)
dictFake = corpora.Dictionary(fakeNews)
docFake = [dictFake.doc2bow(text) for text in fakeNews]


# In[ ]:


# generate coherence scores to determine an optimum number of topics
coherenceValues = []
mdlLst = []

minTopics = 2
maxTopics = 11

for numTopics in range(minTopics, maxTopics+1):
    model = gensim.models.LdaModel(docFake, num_topics=numTopics, id2word = dictFake)
    mdlLst.append(model)
    coherenceMdl = CoherenceModel(model=model, texts=fakeNews, dictionary=dictFake, coherence='c_v')
    coherenceValues.append(coherenceMdl.get_coherence())
    
plt.plot(range(minTopics, maxTopics+1), coherenceValues)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


# create lda model
nTopics = 8 

ldaMdl = gensim.models.LdaModel(corpus=docFake,
                                       id2word=dictFake,
                                       num_topics=nTopics)

ldaMdl.print_topics(num_topics=nTopics, num_words=10)


# In[ ]:





# In[ ]:


def tfidfCorpus(termMatrix):
    # create a corpus using tfidf vectorization
    tfidf = TfidfModel(corpus=termMatrix, normalize=True)
    corpusTfidf = tfidf[termMatrix]
    return corpusTfidf


def getCoherScores(corpus, dictionary, text, minTopics, maxTopics):
    # generate coherence scores to determine an optimum number of topics
    coherenceValues = []  # Initialize coherenceValues here
    mdlLst = []
    for numTopics in range(minTopics, maxTopics + 1):
        model = LsiModel(corpus, num_topics=numTopics, id2word=dictionary, random_seed=0)
        mdlLst.append(model)
        coherenceModel = CoherenceModel(model=model, texts=text, dictionary=dictionary, coherence='c_v')
        coherenceValues.append(coherenceModel.get_coherence())

    # plot results
    plt.plot(range(minTopics, maxTopics + 1), coherenceValues)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(["Coherence Values"], loc='best')
    plt.show()


# Create tfidf representation
corpusTfidfFake = tfidfCorpus(docFake)

# Coherence scores for fake news data
getCoherScores(corpusTfidfFake, dictFake, fakeNews, minTopics=2, maxTopics=11)


# In[ ]:


# model for fake news data
lsa = LsiModel(corpusTfidfFake, id2word=dictFake, num_topics=5)
lsa.print_topics()


# In[ ]:


X = [','.join(map(str, l)) for l in data['text_clean']]
Y = data['fake_or_factual']


# In[ ]:


countvec = CountVectorizer()


# In[ ]:


cvfit = countvec.fit_transform(X)
bow = pd.DataFrame(cvfit.toarray(), columns=countvec.get_feature_names_out())


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(bow, Y, test_size=0.3)


# In[ ]:


lr = LogisticRegression(random_state=0).fit(Xtrain, ytrain)


# In[ ]:


ypredlr = lr.predict(Xtest)


# In[ ]:


accuracy_score(ypredlr, ytest)


# In[ ]:


print(classification_report(ytest, ypredlr))


# In[ ]:


svm = SGDClassifier().fit(Xtrain, ytrain)
ypredsvm = svm.predict(Xtest)
accuracy_score(ypredsvm, ytest)


# In[ ]:


print(classification_report(ytest, ypredsvm))


# In[ ]:




