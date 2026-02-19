"""Train regressors for future subsystem RUL prediction."""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RUL_MODEL_SPECS = {
    "engine": ("engine_rul_pct", "engine_rul_pct_future"),
    "brake": ("brake_rul_pct", "brake_rul_pct_future"),
    "battery": ("battery_rul_pct", "battery_rul_pct_future"),
}


def _train_single_target(df: pd.DataFrame, feature: str, target: str) -> GradientBoostingRegressor:
    X = df[[feature]]
    y = df[target]

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
    print(f"{target} MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"{target} R2: {r2_score(y_test, predictions):.4f}")
    return model


def train_rul_model(data_path: str = "cloud_health_history.csv", model_path: str = "rul_model.pkl") -> None:
    df = pd.read_csv(data_path)

    models: dict[str, GradientBoostingRegressor] = {}
    for name, (feature, target) in RUL_MODEL_SPECS.items():
        models[name] = _train_single_target(df, feature=feature, target=target)

    joblib.dump(models, model_path)
    print(f"Saved model to {model_path}")


def predict_future_rul(models: dict, engine_rul_pct: float, brake_rul_pct: float, battery_rul_pct: float) -> dict[str, float]:
    """Predict future RUL percentages using only corresponding current subsystem RUL."""

    raw = {
        "engine": float(models["engine"].predict(pd.DataFrame([{"engine_rul_pct": engine_rul_pct}]))[0]),
        "brake": float(models["brake"].predict(pd.DataFrame([{"brake_rul_pct": brake_rul_pct}]))[0]),
        "battery": float(models["battery"].predict(pd.DataFrame([{"battery_rul_pct": battery_rul_pct}]))[0]),
    }
    return {k: max(0.0, min(100.0, v)) for k, v in raw.items()}


if __name__ == "__main__":
    train_rul_model()
