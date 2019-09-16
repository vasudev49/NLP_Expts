# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:41:49 2019

@author: vasudeva.maiya
"""


#----------------------------------------------------------------------------
#LDA
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
#--------------------------------------------

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora, models
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import conll2000
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

Path='C:/Users/vasudeva.maiya/Downloads/Dov/ESG/WM'
os.chdir(Path)
stop = (stopwords.words('english'))
#extend the stopwords set - its just a list
stop.remove('yourselves')
stop.extend(['from', 'subject', 're', 'edu', 'use'])
#stop.add(['from', 'subject', 're', 'edu', 'use'])

exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

d = pd.read_excel('FEL 3 yr - 1 of 3.xlsx')
d=d.loc[:,['JOB_ID','JOB_SYSTEM','JOB_TYPE','JOB_NOTE','COMPLAINT_NOTE','CAUSE_NOTE', 'CORRECTION_NOTE']]
#if null then replace with empty string.
d['JOB_NOTE'].fillna('',inplace=True) #.isnull().sum()
jobNote=d['JOB_NOTE'].tolist()

jobNoteCleaned=[clean(doc).split() for doc in jobNote ]
jobNoteCleaned
for note in jobNoteCleaned[0:2]:
    print (note)


# Build the bigram and trigram phrases:
#https://www.pydoc.io/pypi/gensim-3.2.0/autoapi/models/phrases/index.html#module-models.phrases
#https://radimrehurek.com/gensim/models/phrases.html

bigram = gensim.models.phrases.Phrases(jobNoteCleaned, min_count=2, threshold=10) # higher threshold fewer phrases.
#phraser is a wrapper around Phrases which makes the model faster as it discards useless things
bigram=gensim.models.phrases.Phraser(bigram)

#for phrase in bigram[jobNoteCleaned[1:20]]:
#    print(phrase)

trigram = gensim.models.Phrases(bigram[jobNoteCleaned], min_count=10, threshold=100)  
trigram =gensim.models.phrases.Phraser(trigram) 

#for phrase in trigram[bigram[jobNoteCleaned[1:20]]]:
#    print(phrase)

jobNoteCleanedTrigram = [trigram[bigram[doc]] for doc in jobNoteCleaned]
dictionary = corpora.Dictionary(jobNoteCleanedTrigram )
print(dictionary.token2id)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in jobNoteCleanedTrigram]
doc_term_matrix=doc_term_matrix[0:1000]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
ldamodel.print_topics(10)

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus=doc_term_matrix , dictionary=dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
pyLDAvis.show(lda_display)
#pyLDAvis.save_html(lda_display, 'lda.html')

#-----------------------------------
#finding ideal number of topics
#-----------------------------------  
from gensim.models import CoherenceModel
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,passes=30)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, 
                                                        corpus=doc_term_matrix[1:1000], texts=jobNoteCleanedTrigram, 
                                                        start=2, limit=10, step=1)
# Show graph
limit=10; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#-------------------------------------------
#Show Dominant topic in each doc:
#------------------------------------------
def format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix[1:100], texts=jobNoteCleanedTrigram):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
    

df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix[1:100], texts=jobNoteCleanedTrigram)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

#----------------------------------------
# MOst representative Doc for each Topic
#----------------------------------------
# Group top 5 sentences under each topic
sent_topics_sorteddf = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf.head()


#-------------------------------------
# Topic distribution across documents
# ------------------------------------
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)
# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
# Show
df_dominant_topics