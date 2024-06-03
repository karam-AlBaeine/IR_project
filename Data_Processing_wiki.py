
import string
import re
import nltk
from Abbreviations_Wiki import abbreviations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Additional_StopWords_Wiki import additional_stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt')

# convert to lower case 
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

# Remove Punctuation Function
def remove_punctuation(text):
    """
    this method to remove any punctuation in the string for example:

    >>>remove_punctuation("Hi I'm karam! do u have any question? ...,?#@")
    >>>Hi Im karam do u have any question

    """
    text_without_punctuation=''.join([character for character in text if character not in string.punctuation])
    return text_without_punctuation
# remove punctuation test
# print(remove_punctuation("Hi I'm karam! do u have any question? ...,?#@"))


# Rephrasing Abbreviations Function
def rephrasing_abbreviations(text):
    """
    This method returns the shortcuts for the basic sentences for example:
    >>>rephrasing_abbreviations("Hi I'm karam")
    >>>Hi I am karam

    """
    for abbreviation, basic_sentence in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(abbreviation), basic_sentence, text)
    return text
# rephrasing abbreviations test
# print(rephrasing_abbreviations("Hi I'm karam"))


# Data Cleaning Function
def data_cleaning(text):
 """
 This method return the Text after cleaning Data for example :
 >>>data_cleaning("heeeellllloooo now we are cleaning cleaning owr Data DB 100 199 sopendsangsdknasjdnnjnsdnfjanasdfn & return the cleaning data ... ")
 >>>heelloo now  are cleaning owr Data      and  return the cleaning data ... 

 """

 # REmove URLs
 text = re.sub(r"http\S+", "", text)

 # Delete the letter repeated more than twice in the word
 text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
 
 # Delete the duplicate word 
 text = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text)

 # Convert symbols to text
 text = re.sub('\$', " dollar ", text)

 text = re.sub('\%', " percent ", text)

 text = re.sub('\&', " and ", text)

 text = re.sub('\|', " or ", text)

 # Remove not useful numbers
 text = re.sub(r'\b\d+\b', '', text)

 # Remove words that are two characters or shorter
 text = re.sub(r'\b\w{1,2}\b', '', text)

 # Remove words that are longer than twenty characters
 text = re.sub(r'\b\w{21,}\b', '', text)

 return text
# Data Cleaning test
# print(data_cleaning("heeeellllloooo now we are cleaning cleaning owr Data DB 100$ 199| & % sopendsangsdknasjdnnjnsdnfjanasdfn & return the cleaning data ... "))


# Tokenize text Function
def tokenize_text(text):
    """
    This method split the text into tokens for example:
    >>>tokenize_text("Hi I'm karam! do u have any question? ...,?#@")
    >>>['Hi', 'I', "'m", 'karam', '!', 'do', 'u', 'have', 'any', 'question', '?', '...', ',', '?', '#', '@']

    """
    tokens = word_tokenize(text)
    return tokens
# Tokenize text test
# print(tokenize_text("Hi I'm karam! do u have any question? ...,?#@"))


# Remove Stopwords Function
def remove_stopwords(tokens):
    """
    This method remove the stop words from the sentince for example:
    >>>data = ['Hi', 'I', "am", 'karam', '!', 'do', 'u', 'have', 'any', 'question', '?', '...', ',', '?', '#', '@']
    >>>remove_stopwords(data)
    >>>['Hi', 'karam', '!', 'u', 'question', '?', '...', ',', '?', '#', '@']

    """

    filtered_text = []
    for word in tokens:
        if word.lower() not in stopwords.words('English') and word.lower() not in additional_stopwords:
            filtered_text.append(word)
        
    return filtered_text
# Remove Stopwords test
# data = ['Hi', 'I', "am", 'karam', '!', 'do', 'u', 'have', 'any', 'question', '?', '...', ',', '?', '#', '@']
# print(remove_stopwords(data))


# Stemming Function
def stemming(tokens):
    """
    This method removes prefixes and suffixes from words for example:
    >>>stemming( ['Hi', 'I', "played", 'running', 'have', 'any', 'question', 'went','poys','prefixes'])
    >>>['hi', 'i', 'play', 'run', 'have', 'ani', 'question', 'went', 'poy', 'prefix']

    """
    stemmer = PorterStemmer()
    data_after_stemming = [stemmer.stem(word) for word in tokens]
    return data_after_stemming
# Stemming test
# print(stemming( ['Hi', 'I', "played", 'running', 'have', 'any', 'question', 'went','poys','prefixes']))


def get_wordnet_pos(tag_parameter):

    tag = tag_parameter[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatization Function
def data_lemmatization(tokens):
    """
    This method returns the word to its correct linguistic root for example:
    >>>data_lemmatization(['Hi', 'I', "played", 'running','am','was' ,'have', 'any', 'question', 'went','poys','prefixes']
    >>>['Hi', 'I', 'play', 'run', 'be', 'be', 'have', 'any', 'question', 'go', 'poys', 'prefix']

    """
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized_words
# Data Lemmatization test
# print(data_lemmatization(['Hi', 'I', "played", 'running','am','was' ,'have', 'any', 'question', 'went','poys','prefixes']))


# Data Processing Function
def data_processing_wiki(text):
  """
  This method calls data cleaning methods in a specific order and returns the text after performing operations on it for example:
  >>>data_processing('Hi I am Karam , now we are Testing the Data processssingg function , here we havr the misstak in the textt , do you have any problems ? IN USA UK')
  >>>karam test data process function mistak text problem unit state america unit kingdom
  
  """
  text = remove_punctuation(text)
  text = rephrasing_abbreviations(text)
  text = data_cleaning(text)
  tokens = tokenize_text(text)
  tokens = remove_stopwords(tokens)
  tokens = stemming(tokens) 
  tokens = to_lowercase(tokens)
  tokens = data_lemmatization(tokens)

  return " ".join(tokens)  
# Data Processing test
# print(data_processing('Hi I am Karam , now we are Testing the Data processssingg function , here we havr the misstak in the textt , do you have any problems ? IN USA UK'))


######################################################## wiki dataset porcessing ########################################################
# import pandas as pd

# data = pd.read_csv('wiki.csv')
# data = data.fillna('defualt value')
# idx = data['id_right'].tolist()
# texts = data['text_right'].tolist()
# process_corpus = {id : data_processing_wiki(str(text)) for id ,text in zip(idx,texts)}
# df = pd.DataFrame({'text': process_corpus})
# df.to_csv('cleaned_wiki.csv', index=False)