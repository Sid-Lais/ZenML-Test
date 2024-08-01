from zenml import pipeline, step
from random import randint

# Step to train the model
@step
def training_model(model: str) -> int:
    accuracy = randint(1, 10)
    print(f"Model {model} trained with accuracy: {accuracy}")
    return accuracy

# Step to choose the best model
@step
def choosing_best_model(training_model_A: int, training_model_B: int, training_model_C: int) -> str:
    accuracies = [training_model_A, training_model_B, training_model_C]
    print(f"Accuracies: {accuracies}")
    if max(accuracies) > 8:
        return 'accurate'
    return 'inaccurate'

# Step for accurate result
@step
def accurate() -> None:
    print("accurate")

# Step for inaccurate result
@step
def inaccurate() -> None:
    print("inaccurate")

# Define the ZenML pipeline
@pipeline
def zenml_pipeline():
    model_A = training_model(model='A')
    model_B = training_model(model='B')
    model_C = training_model(model='C')

    best_model_decision = choosing_best_model(
        training_model_A=model_A,
        training_model_B=model_B,
        training_model_C=model_C
    )

    if best_model_decision == 'accurate':
        accurate()
    else:
        inaccurate()

if __name__ == "__main__":
    # Run the pipeline
    zenml_pipeline()

