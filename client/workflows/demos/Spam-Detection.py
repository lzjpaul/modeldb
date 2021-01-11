#!/usr/bin/env python
# coding: utf-8

# # Spam Detection (NLTK and scikit-learn)

# #### This example logs a `class` (instead of an object instance) as a model.
# This allows for custom setup configuration in the class's `__init__()` method,  
# and access to logged artifacts at deployment time.

# In[1]:
from __future__ import print_function

try:
    import verta
except ImportError:
    get_ipython().system('pip install verta')


# This example features:
# - word similarity detection using [WordNet](https://github.com/nltk/wordnet) from **NLTK**
# - [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectorization using **scikit-learn**
# - **verta**'s Python client logging a `class` as a model to be instantiated at deployment time
# - predictions against a deployed model

# In[2]:


# HOST = "app.verta.ai"
HOST = "http://localhost:3009"

PROJECT_NAME = "Spam Detection"
EXPERIMENT_NAME = "tf–idf"


# In[3]:


# import os
# os.environ['VERTA_EMAIL'] = 
# os.environ['VERTA_DEV_KEY'] = 


# ## Imports

# In[4]:


# from __future__ import print_function

import json
import os
import re
import time

import cloudpickle

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from IPython.display import display

# In[5]:


try:
    import wget
except ImportError:
    get_ipython().system('pip install wget  # you may need pip3')
    import wget


# ---

# # Run Workflow

# ## Prepare Data

# In[6]:


train_data_url = "http://s3.amazonaws.com/verta-starter/spam.csv"
train_data_filename = wget.detect_filename(train_data_url)
if not os.path.isfile(train_data_filename):
    wget.download(train_data_url)


# In[7]:


raw_data = pd.read_csv(train_data_filename, delimiter=',', encoding='latin-1')

raw_data.head()


# In[8]:


# turn spam/ham to 0/1, and remove unnecessary columns
raw_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
raw_data.v1 = LabelEncoder().fit_transform(raw_data.v1)

raw_data.head()


# In[9]:


# lemmatize text
total_stopwords = set([word.replace("'",'') for word in stopwords.words('english')])
lemma = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.replace("'",'')
    text = re.sub('[^a-zA-Z]',' ',text)
    words = text.split()
    words = [lemma.lemmatize(word) for word in words if (word not in total_stopwords) and (len(word)>1)] # Remove stop words
    text = " ".join(words)
    return text

raw_data.v2 = raw_data.v2.apply(preprocess_text)

raw_data.head()


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(raw_data.v2, raw_data.v1, test_size=0.15, stratify=raw_data.v1)    


# ## Instantiate Client

# In[11]:


from verta import Client
from verta.utils import ModelAPI

client = Client(HOST)
proj = client.set_project(PROJECT_NAME)
expt = client.set_experiment(EXPERIMENT_NAME)
run = client.set_experiment_run()


# ## Fit Model

# In[12]:


vectorizer = TfidfVectorizer()
vectorizer.fit(x_train)

x_train_vec = vectorizer.transform(x_train).toarray()

model = linear_model.LogisticRegression()
model.fit(x_train_vec, y_train)


# In[13]:


x_test_vec = vectorizer.transform(x_test).toarray()
y_pred = model.predict(x_test_vec)

m_confusion_test = confusion_matrix(y_test, y_pred)
display(pd.DataFrame(data=m_confusion_test,
                     columns=['Predicted 0', 'Predicted 1'],
                     index=['Actual 0', 'Actual 1']))

print("This model misclassifies {} genuine SMS as spam"
      " and misses only {} SPAM.".format(m_confusion_test[0,1], m_confusion_test[1,0]))


# In[14]:


accuracy = accuracy_score(y_test, y_pred)

run.log_metric("accuracy", accuracy)

accuracy


# In[15]:


# save and upload weights
model_param = {}
model_param['coef'] = model.coef_.reshape(-1).tolist()
model_param['intercept'] = model.intercept_.tolist()

json.dump(model_param, open("weights.json", "w"))

run.log_artifact("weights", open("weights.json", "rb"))


# In[16]:


# serialize and upload vectorizer
run.log_artifact("vectorizer", vectorizer)


# ## Define Model Class

# Our model—with its pre-trained weights and serialized vectorizer—will require some setup at deployment time.
# 
# To support this, the Verta platform allows a model to be defined as a `class` that will be instantiated when it's deployed.  
# This class should have provide the following interface:
# 
# - `__init__(self, artifacts)` where `artifacts` is a mapping of artifact keys to filepaths. This will be explained below, but Verta will provide this so you can open these artifact files and set up your model. Other initialization steps would be in this method, as well.
# - `predict(self, data)` where `data`—like in other custom Verta models—is a list of input values for the model.

# In[17]:


class SpamModel():    
    def __init__(self, artifacts):
        from nltk.corpus import stopwords  # needs to be re-imported to remove local file link
        
        # get artifact filepaths from `artifacts` mapping
        weights_filepath = artifacts['weights']
        vectorizer_filepath = artifacts['vectorizer']

        # load artifacts
        self.weights = json.load(open(weights_filepath, "r"))
        self.vectorizer = cloudpickle.load(open(vectorizer_filepath, "rb"))
        
        # reconstitute logistic regression
        self.coef_ = np.array(self.weights["coef"])
        self.intercept_ = self.weights["intercept"]
        
        # configure text preprocessing
        self.total_stopwords = set([word.replace("'",'') for word in stopwords.words('english')])
        self.lemma = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = text.lower()
        text = text.replace("'",'')
        text = re.sub('[^a-zA-Z]',' ',text)
        words = text.split()
        words = [self.lemma.lemmatize(word) for word in words if (word not in self.total_stopwords) and (len(word)>1)] # Remove stop words
        text = " ".join(words)
        return text     
        
    def predict(self, data):
        predictions = []
        for inp in data:
            # preprocess input
            processed_text = self.preprocess_text(inp)
            inp_vec = self.vectorizer.transform([inp]).toarray()
            
            # make prediction
            prediction = (np.dot(inp_vec.reshape(-1), self.coef_.reshape(-1)) + self.intercept_)[0]
            predictions.append(prediction)
            
        return predictions


# Earlier we logged artifacts with the keys `"weights"` and `"vectorizer"`.  
# You can obtain an `artifacts` mapping mentioned above using `run.fetch_artifacts(keys)` to work with locally.  
# A similar mapping—that works identically—will be passed into `__init__()` when the model is deployed.

# In[18]:


artifacts = run.fetch_artifacts(["weights", "vectorizer"])

spam_model = SpamModel(artifacts=artifacts)


# In[19]:


spam_model.predict(["FREE FREE FREE"])


# ## Log Model

# The keys expected in the `artifacts` mapping mentioned above must be passed into `run.log_model()` to be available during deployment!

# In[20]:


run.log_model(
    model=SpamModel,
    artifacts=['weights', 'vectorizer'],
)


# We also have to make sure we provide every package involved in the model.

# In[21]:


run.log_requirements([
    "cloudpickle",
    "nltk",
    "numpy",
    "sklearn",
])


# And we need to ensure that the appropriate NLTK packages are available during deployment.

# In[22]:


run.log_setup_script("""
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
""")


# ---

# # Make Live Predictions

# Access the Experiment Run through the Verta Web App and deploy it!

# In[23]:


run


# ## Load Deployed Model

# In[24]:


from verta.deployment import DeployedModel

deployed_model = DeployedModel(HOST, run.id)


# ## Query Deployed Model

# In[25]:


for text in x_test:
    print(deployed_model.predict([text]))
    time.sleep(.5)


# ---
