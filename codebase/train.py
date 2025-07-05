from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib


reviews = load_files("../data/imdb", categories=["pos", "neg"])
X, y = reviews.data, reviews.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'clf__C': [0.1, 1, 10]
}
mlflow.set_experiment("sentiment_analysis")
with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=3)
    grid.fit(X_train, y_train)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", grid.best_score_)

    joblib.dump(grid.best_estimator_, "../codebase/model.joblib")
    mlflow.sklearn.log_model(grid.best_estimator_, "model")
