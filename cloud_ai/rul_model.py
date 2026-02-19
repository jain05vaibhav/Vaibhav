"""Train Gradient Boosting regressor for future engine RUL prediction."""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RUL_FEATURES = [
    "engine_rul_pct",
    "brake_rul_pct",
    "battery_rul_pct",
]
RUL_TARGET = "engine_rul_pct_future"


def train_rul_model(data_path: str = "cloud_health_history.csv", model_path: str = "rul_model.pkl") -> None:
    df = pd.read_csv(data_path)

    X = df[RUL_FEATURES]
    y = df[RUL_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"RUL MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RUL R2: {r2_score(y_test, predictions):.4f}")

    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_rul_model()
