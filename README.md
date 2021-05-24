# Applications-of-Data-Science

- This repository contains programming assignments from the core subject `Applications of Data Science`, that were completed as part of the `Plaksha Tech Leaders Fellowship` program.

### Table of Contents
1. [Instructions to use the repository](#instructions)
2. [List of Dependencies](#dependency)
3. [File Descriptions](#desc)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## List of Dependencies<a name="dependency"></a>
The `requirements file` list all the libraries/dependencies required to run the notebooks/projects in this repository.

## Instructions to use the repository<a name="instructions"></a>
1. Clone this github repository and start using it.
`git clone https://github.com/Ankit-Kumar-Saini/Applications-of-Data-Science`


## File Descriptions<a name="desc"></a>
1. `Sentiment Analysis folder`: It contains code to deploy a web app for movie review sentiment analysis. The following files/folders are present inside
	1. app folder: It contains the main python script `app.py` to run the web app, templates to render the web app and a `SQL database` to store the user inputs.
	2. models folder: It contains the tokenizer and weights of the model. These are loaded to run the model for making predictions on user input during inference.
	3. sentiment analysis.ipynb: This jupyter notebook contains the code to train the model using bag of words approach on movie reviews. It also has the code to `train Word Embeddings` from scratch.

2. Basics of SQL.ipynb: This jupyter notebook demonstrates the use of basic SQL commands on a toy dataset.

3. Transfer Learning.ipynb: This notebook contains code to classify images of Cat, Kanye West and Pickachu using pre-trained EfficientNetB0 model.

4. WebScrapping1.ipynb: This notebook demonstrates the use of `Beautiful Soup` library for web scrapping with an example.

5. WebScrapping2.ipynb: This notebook uses `Beautiful Soup` library to scrape Covid-19 data from Wikipedia. Exploratory data analysis (EDA) is carried out on this data and a linear regression model is trained to make predictions on the number of covid-19 cases for selected countries.

6. Visualization data: This folder contains data for creating visualizations in `matplotlib`.

7. dog_cat: This is a dummy database file created to demonstrate the use of basic SQL queries.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Plaksha for the data and python 3 notebook.




