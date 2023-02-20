# Sentiment Analysis for Amazon Reviews

## Project Overview
The purpose of this project is to create a product classifier that predicts whether a person likes a product or not. To accomplish this task, we utilized traditional machine learning models such as Logistic Regression and Support Vector Machines (SVMs). We used a dataset of customer reviews and ratings to train and test our models. 


## Requirements

- Python 3
- scikit-learn
- pandas
- numpy


## Data Description

The dataset used in this project was obtained from Kaggle [link](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset). The dataset contains customer reviews and ratings for a variety of products on Amazon. To label the data, we considered a rating of 3.5 or more as a good review and a rating less than 3.5 as a bad review. To vectorize the data, we utilized the bag-of-words method to represent each review as a vector of word counts.




## Instructions
To reproduce the project, please install the packages listed in the yaml file.
```
conda env create -f nlp.yml
conda activate nlp
```


## Results 
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.954 |
| Support Vector Classifier | 0.93 |



## Future work
In future projects, I plan to incorporate an extensive data collection pipeline to gather more diverse and high-quality data. Additionally, I aim to leverage large language models to improve my current results. This could involve exploring pre-trained models like GPT-3 and fine-tuning them on our specific task. By doing so, I hope to create more robust and accurate models that can better generalize to new and unseen data.