Titanic – Data Science Solutions (with Cabin feature)
This repo extends the classic “Titanic: Data Science Solutions” notebook by adding:
1.a CabinDeck feature engineered from the raw Cabin column, and
2.feature scaling for continuous variables.
These changes improved several model accuracies compared to the original baseline.

What’s in this repo
titanic-data-science-solutions-with-cabin-feature.ipynb
A fork/derivative of the popular Kaggle notebook. I annotated the new/modified cells with clear headings like "+ Added: CabinDeck feature” and "+ Added: Scaling”.

Key changes (what I did and where)
1) Keep signal from Cabin via CabinDeck
Where: Early in the feature-engineering section, right before dropping columns.
What & Why:
Extract the first character of Cabin (deck A–G, T; unknown → U).
Encode it consistently across train/test (fit the map on the union of both).
Drop the original sparse Cabin string after extraction.
Rationale: Deck location captures passenger position on the ship, which is plausibly related to lifeboat access/survival.

Code to check :
# Create CabinDeck BEFORE dropping Cabin
for df in (train_df, test_df):
    df['CabinDeck'] = df['Cabin'].str[0].fillna('U')

# Drop Cabin (and Ticket if desired)
train_df = train_df.drop(columns=['Cabin', 'Ticket'])
test_df  = test_df.drop(columns=['Cabin', 'Ticket'])

# Encode CabinDeck consistently across train and test
deck_all = pd.concat([train_df['CabinDeck'], test_df['CabinDeck']])
deck_map = {d:i for i,d in enumerate(sorted(deck_all.unique()))}
for df in (train_df, test_df):
    df['CabinDeck'] = df['CabinDeck'].map(deck_map).astype(int)


2) Scale continuous features

Where: After feature engineering and before model training.
What & Why:
Standardize (StandardScaler) the continuous columns to mean 0 / std 1.
Helps distance/gradient-based models (KNN, Perceptron, SVM); trees are scale-invariant.
Columns scaled: Age, Fare, Age*Class.

Code to check :
from sklearn.preprocessing import StandardScaler
scale_cols = ['Age', 'Fare', 'Age*Class'] 
scaler = StandardScaler()
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
test_df[scale_cols]  = scaler.transform(test_df[scale_cols])

Takeaways
1.CabinDeck adds meaningful signal → strong lift for tree models and KNN.
2.Scaling further boosts KNN and Perceptron as expected.
3.Trees remain unchanged with scaling (as expected).



How to run
1.Environment
  Python 3.x
  pandas, numpy, scikit-learn, matplotlib, seaborn
2.Data
  Use Kaggle’s Titanic dataset (train.csv, test.csv).
  If running on Kaggle: the dataset is pre-mounted at /kaggle/input/titanic/.
  If running locally: place CSVs in a data/ folder and adjust the paths near the top of the notebook.

3.Notebook
  Open titanic-data-science-solutions-with-cabin-feature.ipynb.
  Run all cells. The results tables will be printed at the end.

What stays from the original notebook
  Title extraction from Name (with rare-title grouping).
  FamilySize and IsAlone from SibSp + Parch.
  Imputation and encoding for Age, Fare, Embarked, Sex.
  Interaction Age*Class.
  Model suite: SVM, KNN, Logistic Regression, Random Forest, Naive Bayes, Perceptron, SGD, Linear SVC, Decision Tree.


