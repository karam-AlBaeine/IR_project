# IR-Project
# Information Retrieval System

This is an information retrieval system built with Python Flask using a Service Oriented Architecture (SOA). It uses Sqlite3 as the database to save History and HTML & CSS & jquery & bootstrap for style , We use TF_IDF to indexing from sklearn library .

## Project Files

The project is structured as follows:

-app file have a routes for run all files and call the functions from auther files.

-Data_Processing files to process data set .

-Data_Representaion to build tf_idf matrix and vectorizer .

-Indexing to save and load the index .

-Query_Refinment to customize queries and corect it .

-Query_processing to build a vector for query after process it .

-Evaluation to calculate evaluation using Pression@k & ReCall@k & MAP & MRR.

-personalization to match queries using person favorates.

-Summarizing to summarize text 

-Templets have a html pages.

## Features

- Personalization
- SOA (Service Oriented Architecture)
- Auto Summarizing
- ChatPot 
- LDA Model from sklearn.decomposition

## Requirements

- Python 3.x
- Flask
- Nltk
- tensorformers

## Installation

### Clone the Repository

git clone https://github.com/karam-AlBaeine/IR-Project

### Set Up Virtual Environment

python -m venv venv
venv\Scripts\activate

### Install Dependencies

Install Requirements using pip install in your Virtual Environment


