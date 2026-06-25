from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FailurePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50)

        # dummy training for now (so predict works immediately)
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        self.model.fit(X, y)

    def predict(self, metrics):
        features = [[
            metrics['cpu'],
            metrics['mem'],
            metrics['latency'],
            metrics['errors']
        ]]
        return self.model.predict_proba(features)[0][1]
