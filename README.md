# Predictive Product Categorizer 🤖🛒


This project is an AI/ML text classification model that automatically categorizes e-commerce product descriptions into predefined categories. It was built in a Jupyter Notebook (Google Colab) and demonstrates the full pipeline of data preprocessing, feature engineering, model training, and evaluation.

The goal is to solve the challenge of manual product categorization, which is time-consuming, by building a machine learning model that learns patterns from product descriptions and predicts their correct category.


---

## Dataset
The model uses a text dataset sourced from Kaggle.  


Categories: Multiple e-commerce categories such as electronics, clothing and accessories, household items,books

Features: Each record contains a product description (text) and its corresponding category (label)



📂 [E-commerce Text Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)



---

## Project Files
- **Text_Classification_Model.ipynb**: The main Jupyter Notebook file containing all the code and analysis.
**requirements.txt** – Python libraries needed to run the notebook

**README.md** – Project documentation



---

## Key Features
- **Data Preprocessing**  
  Handles raw text data, including tokenization, stop word removal, and other text cleaning operations using *nltk*.

- **Feature Engineering**  
  Converts text data into a numerical format using a TF-IDF Vectorizer to prepare it for model training.

- **Model Training**  
  Trains a Linear Support Vector Classifier (*LinearSVC*) from scikit-learn for the classification task.

- **Model Evaluation**  
  Assesses model performance using a classification report and a confusion matrix, which are essential for understanding the model's accuracy, precision, and recall across different classes.

---

## Libraries Used
This project makes use of the following popular Python libraries:

- matplotlib  
- nltk  
- numpy  
- pandas  
- scikit-learn  
- seaborn  

---

## Model Details
- Uses `sklearn.model_selection` for splitting data into training and testing sets.  
- Employs **StratifiedKFold** for robust cross-validation, ensuring the model's performance is stable and reliable.  
- Achieves high accuracy and strong performance metrics on training data.  


## How to Run
This project is designed to be run directly on **Google Colab**.

1. Upload the `Text_Classification_Model.ipynb` file to your Google Colab environment.  
2. Run the cells sequentially to execute the full workflow, from data loading to model evaluation.  

---

## Conclusion & Future Improvements

This project demonstrates how machine learning can automate product categorization in e-commerce, reducing manual effort and improving accuracy.

Learnings:

Preprocessing text is critical for model performance

TF-IDF + LinearSVC is a strong baseline for text classification

Cross-validation improves trust in the model’s results


Future improvements could include:

Trying deep learning models (e.g., LSTMs, BERT) for even higher accuracy

Hyperparameter tuning to optimize the LinearSVC model

Expanding dataset size for better generalization


