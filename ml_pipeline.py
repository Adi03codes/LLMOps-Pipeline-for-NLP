import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

mlflow.start_run()

# Load data
df = pd.read_csv("data/news.csv")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Metrics
preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)
print(report)

# Log model
mlflow.sklearn.log_model(model, "log_reg")
mlflow.end_run()
