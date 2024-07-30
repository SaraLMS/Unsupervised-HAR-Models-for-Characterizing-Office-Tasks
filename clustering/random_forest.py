from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from constants import CLASS, SUBCLASS
import load


def random_forest_classifier(x_train, x_test, y_train, y_test):

    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=1)

    rf.fit(x_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(x_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
