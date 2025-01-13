#!/usr/bin/env python
# coding: utf-8

# # Search Code
# ___

# In[1]:


#Library Import
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from FlagEmbedding import FlagModel

#Download from nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# #### Pre-Process

# In[2]:


class TextProcessor:
    def __init__(self, paths):
        #Initialize resources
        self.ENGLISH_WORDS = set(words.words())
        self.lemmatizer = WordNetLemmatizer()
        self.STOP_WORDS = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.paths = paths
        self.database = []
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(min_df=2)
    
    def is_english_word(self, word):
        return word.lower() in self.ENGLISH_WORDS

    def processed_documents(self, query):
        """Process query text: lemmatize, remove stopwords, and stem."""
        words_list = query.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words_list]
        filtered = [word for word in lemmatized if word not in self.STOP_WORDS and self.is_english_word(word)]
        stemmed = [self.stemmer.stem(word) for word in filtered]
        return " ".join(stemmed)
    
    def process_query(self, query):
        """Process query text: lemmatize, remove stopwords, and stem."""
        words_list = query.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words_list]
        filtered = [word for word in lemmatized if word not in self.STOP_WORDS and self.is_english_word(word)]
        stemmed = [self.stemmer.stem(word) for word in filtered]
        return " ".join(stemmed)
    
    def load_database(self):
        """Load .pkl files and extract KEYWORD entries."""
        pickle_folder = self.paths["windows"] if os.name == "nt" else self.paths["macos"]
        pkl_file_count = 0
        for root, dirs, files in os.walk(pickle_folder):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_file_count += 1
                    with open(os.path.join(root, file), "rb") as f:
                        self.database.extend(pickle.load(f))
        print(f"Number of .pkl files: {pkl_file_count}")
    
    def vectorize_database(self):
        """Vectorize the text data in the database using TF-IDF."""
        documents = [entry['KEYWORD'] for entry in self.database if 'KEYWORD' in entry]
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
    
    def get_tfidf_matrix(self):
        return self.tfidf_matrix


# #### TF-IDF Search

# In[3]:


def tfidf_search(query, n, text_processor):
    processed_query = text_processor.process_query(query)
    print(f"\nProcessed Query: {processed_query}")

    #Pre-Process with TextProcessor Class
    if text_processor.tfidf_matrix is None:
        text_processor.load_database()
        text_processor.vectorize_database()

    tfidf_matrix = text_processor.get_tfidf_matrix()
    vectorizer = text_processor.vectorizer

    #Pre-Process the Query Texts
    query_vector = vectorizer.transform([processed_query])

    #Search Result
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    top_n_indices = np.argsort(similarities[0])[::-1][:n]

    top_n_neighbors = [(text_processor.database[i], similarities[0, i]) for i in top_n_indices]

    #Print Result
    print("TF-IDF Search Results: \n")
    for rank, (doc, score) in enumerate(top_n_neighbors, start=1):
        print(f"Rank {rank}: \nSimilarity Score: {score:.4f},\nDOCUMENT_____")
        for key, value in doc.items():
            print(f"{key}: {repr(value)}")

        neighbor_vector = tfidf_matrix[top_n_indices[rank-1]]  # Access the correct vector
        top_indices = neighbor_vector.toarray().flatten().argsort()[::-1]  
        top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_indices[:10]]  
        print("Top Terms:", ", ".join(top_terms))  
        print()


# #### SVD Search

# In[4]:


def svd_search(query, n, text_processor):
    processed_query = text_processor.process_query(query)
    print(f"\nProcessed Query for SVD: {processed_query}")

    #Pre-Process with TextProcessor Class
    if text_processor.tfidf_matrix is None:
        text_processor.load_database()
        text_processor.vectorize_database()

    tfidf_matrix = text_processor.get_tfidf_matrix()
    vectorizer = text_processor.vectorizer

    svd = TruncatedSVD(n_components=100)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    #Pre-Process the Query Texts
    query_vector = vectorizer.transform([processed_query])
    query_svd = svd.transform(query_vector)

    #Search Result
    similarities = cosine_similarity(query_svd, svd_matrix)

    top_n_indices = np.argsort(similarities[0])[::-1][:n]

    top_n_neighbors = [(text_processor.database[i], similarities[0, i]) for i in top_n_indices]

    #Print Result
    print("SVD Search Results: \n")
    for rank, (doc, score) in enumerate(top_n_neighbors, start=1):
        print(f"Rank {rank}: \nSimilarity Score: {score:.4f},\nDOCUMENT_____")
        for key, value in doc.items():
            print(f"{key}: {repr(value)}")
        print()


# #### w2v Search

# In[5]:


def word2vec_search(query, n, text_processor):
    processed_query = text_processor.process_query(query)
    print(f"\nProcessed Query for Word2Vec: {processed_query}")

    #Pre-Process with TextProcessor Class
    if len(text_processor.database) == 0:
        text_processor.load_database()

    #Pre-Process
    model = Word2Vec([doc['KEYWORD'].split() for doc in text_processor.database], min_count=1)

    #Pre-Process the Query Texts
    query_words = [word for word in processed_query.split() if word in model.wv]
    if query_words:
        query_vector = np.mean([model.wv[word] for word in query_words], axis=0)
    else:
        print("No valid words found in the query.")
        return  

    #Search Result
    similarities = []
    for doc in text_processor.database:
        doc_words = [word for word in doc['KEYWORD'].split() if word in model.wv]
        if doc_words:
            doc_vector = np.mean([model.wv[word] for word in doc_words], axis=0)
            similarity = cosine_similarity([query_vector], [doc_vector])
            similarities.append((doc, similarity[0][0]))
        else:
            similarities.append((doc, 0.0))  # If no valid words in doc, similarity is 0

    top_n_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    #Print Result
    print("Word2Vec Search Results: \n")
    for rank, (doc, score) in enumerate(top_n_neighbors, start=1):
        print(f"Rank {rank}: \nSimilarity Score: {score:.4f},\nDOCUMENT_____")
        for key, value in doc.items():
            print(f"{key}: {repr(value)}")
        print()


# #### LLM Search

# In[6]:


def LLM_search(query, n, processor):
    #LLM Model
    model_name = 'BAAI/bge-large-en-v1.5'
    model = FlagModel(
        model_name,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:"
    )

    #Pre-Process the Query Texts
    processed_query = processor.process_query(query)
    print(f"Processed Query: {processed_query}")

    #Pre-Process with TextProcessor Class
    if len(processor.database) == 0:
        processor.load_database()

    #Pre-Process
    embeddings = model.encode([entry['TEXT'] for entry in processor.database])
    query_embedding = model.encode_queries([query])

    #Search Result
    similarities = embeddings @ query_embedding.T
    similarities = similarities.flatten()
    
    top_indices = similarities.argsort()[-n:][::-1]
    
    top_results = [
        {
            **processor.database[i],
            "SIMILARITY": similarities[i]
        }
        for i in top_indices
    ]
    
    #Print Result
    print("LLM Search Results:\n")
    for rank, result in enumerate(top_results, start=1):
        print(f"Rank {rank}:")
        print(f"Similarity Score: {result['SIMILARITY']:.4f},")
        print("DOCUMENT_____")
        print(f"TEXT: {repr(result['TEXT'])}")
        if 'LINE' in result:
            print(f"LINE: {result['LINE']}")
        if 'BOOK' in result:
            print(f"BOOK: {repr(result['BOOK'])}")
        if 'KEYWORD' in result:
            print(f"KEYWORD: {repr(result['KEYWORD'])}")
        if 'Top Terms' in result:
            print(f"Top Terms: {result['Top Terms']}")
        print()

    return


# In[ ]:




