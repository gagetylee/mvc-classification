from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train):
    """
    Trains a random forest classification model

    Parameters
    -----------------------------------
    df: DataFrame containing training data

    Returns
    -----------------------------------
    trained model

    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    return model