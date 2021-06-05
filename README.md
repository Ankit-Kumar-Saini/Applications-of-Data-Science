# Applications of Data Science

- This repository contains programming assignments from the core subject `Applications of Data Science`, that were completed as part of the `Plaksha Tech Leaders Fellowship` program.

### Table of Contents
1. [Instructions to use the repository](#instructions)
2. [List of Dependencies](#dependency)
3. [File Descriptions](#desc)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Instructions to use the repository<a name="instructions"></a>
1. Clone this github repository and start using it.
`git clone https://github.com/Ankit-Kumar-Saini/Applications-of-Data-Science`

## List of Dependencies<a name="dependency"></a>
The `requirements file` list all the libraries/dependencies required to run the notebooks/projects in this repository.

## File Descriptions<a name="desc"></a>
1. `Sentiment Analysis`: This folder contains relevant scripts to deploy a web app for movie review sentiment analysis. The following files/folders are present inside

	1. app: This folder contains the main python script `app.py` to run the web app, templates to render the web app and a `SQL database` to store the user inputs.

	2. models: This folder contains the **CountVectorizer tokenizer** to process the user inputs and weights of the RandomForestClassifier model. These are loaded for making predictions on user input during inference.

	3. `sentiment analysis.ipynb`: This jupyter notebook contains the code to train a **RandomForestClassifier model** using the bag of words approach on movie reviews. It also has the code to **train Word Embeddings** from scratch using **Gensim library**.


2. `Basics of SQL.ipynb`: This jupyter notebook demonstrates the use of basic SQL commands on a toy dataset.


3. `Transfer Learning.ipynb`: This jupyter notebook demonstrates the power of **Transfer Learning** in image classification tasks. Images of Cat, Kanye West and Pickachu were downloaded from the internet. A pre-trained **EfficientNetB0 model** was used as a feature extractor to classify the images into three categories. The model achieved an accuracy of `99.82% on validation data` even with a small training dataset.

4. `Kaggle Cats vs Dogs Classification.ipynb`: This jupyter notebook dives into the interpretation of convolutional neural networks. A pre-trained **InceptionV3 model** was used as a feature extractor to classify images of cats and dogs. **Saliency Maps** of the last convolutional layer were generated to visualize what features are important to classify an image into a cat or dog.

5. `WebScrapping1.ipynb`: This jupyter notebook demonstrates the use of **Beautiful Soup** library for web scrapping with examples.
 

6. `WebScrapping2.ipynb`: This jupyter notebook uses **Beautiful Soup** library to scrape Covid-19 data from Wikipedia. Exploratory data analysis (EDA) was carried out on this data and a linear regression model was trained to make predictions on the number of covid-19 cases for selected countries.


7. Visualization data: This folder contains data for creating visualizations in **matplotlib**.

8. dog_cat: This is a dummy database file created to demonstrate the use of basic SQL queries.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Plaksha for the data and python 3 notebook.




