# Kaggle-Home-Depot-Search-Relevancy
*Machine Learning model to rate Search Relevancy for Home Depot.*

**Visit**:[Kaggle Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance) for more info and data file (could not upload on GitHub because of size)

**Implementation Details**: (using cosine similarity and linear regression)
* Starting by reading all the input files into a pandas dataframe in python. The next step is to
extract the desired information from product_descriptions and attributes file.
* Then merge the required data from product_description and attributes file into train (dataframe
in python)
* The fields used for merging are: product_description (from product_description.csv) and MFG
Brand Name, Material, Bullet1, Bullet2 and Bullet3 (from attributes.csv)
* So, the dataframe train has has these fields: id, product_uid, product_title, search_term,
relevance, product_description, MFG Brand Name, Material, bullet1, bullet2 and bullet3
* Then, dividing the data for each row in train into three parts: one is search_term (store in one
dictionary), second is concatenation of product_description, product_title, MFG Brand Name,
Material, bullet1, bullet2 and bullet3 (another dictionary) and third is the relevance field (kept
for linear regression, not used at this step)
* Implementing the stemming and removing stopwords (after lowering the tokens) on all the
desired fields which are to used in cosine similarity calculation.
* After calculating the cosine similarity between search term and other fields, there are two
columns with numerical data: similarity and relevance.
* Implement Linear Regression on these two where x is similarity and y is relevance and theta
values are to be calculated.
* So, in the end when test data comes, same merging and stemming processing is performed and
cosine similarity is calculated search_term and other fields.
* Then relevance is predicted using theta values from Linear Regression algorithm.


**References:**

*Cosine similarity:*
* http://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
* http://stackoverflow.com/questions/1746501/can-someone-give-an-example-of-cosinesimilarity-in-a-very-simple-graphical-wa

*Linear regression:*
* http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html

*Panda dataframe functions (merge, rename, lambda):*
* http://pandas.pydata.org/pandas-docs/stable/merging.html
* http://chrisalbon.com/python/pandas_apply_operations_to_dataframes.html
* https://gist.github.com/bsweger/e5817488d161f37dcbd2
* http://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe
