from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from constants import CLASS, SUBCLASS
import load


def random_forest_classifier(x_train, x_test, y_train, y_test):

    # Initialize the random forest classifier
    rf = RandomForestClassifier()

    # Set up the hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    # Configure Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    # Fit the model
    grid_search.fit(x_train, y_train)

    # Get the best estimator
    best_rf = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_rf.predict(x_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return accuracy
