from zenml import pipeline, step
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step to load data
@step
def load_data() -> tuple:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step to train the model
@step
def train_model(X_train: list, y_train: list) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step to evaluate the model
@step
def evaluate_model(model: RandomForestClassifier, X_test: list, y_test: list) -> float:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Define the ZenML pipeline
@pipeline
def ml_workflow_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

# Run the ZenML pipeline
if __name__ == "__main__":
    ml_workflow_pipeline()
