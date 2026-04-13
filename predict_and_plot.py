import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def run_inference_and_plot():
    # 1. Load the models
    print("Loading models...")
    try:
        model_engine = joblib.load("engine_rul_model.pkl")
        model_brake = joblib.load("brake_rul_model.pkl")
        model_battery = joblib.load("battery_rul_model.pkl")
        model_failure = joblib.load("failure_model.pkl")
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. {e}")
        return

    # 2. Define sample input feature vector (user's 7 features)
    feature_names_user = [
        "thermal_stress_index",
        "vibration_score",
        "electrical_efficiency",
        "brake_health_index",
        "engine_load",
        "rpm",
        "engine_knock_probability"
    ]
    
    # Sample data values provided by user or defaults
    sample_values = [0.45, 0.22, 0.88, 0.75, 65.0, 2500, 0.05]
    user_input = dict(zip(feature_names_user, sample_values))

    # 3. Mapping to model-expected features
    # RUL models expect: [thermal_stress_index, brake_health_index, mechanical_vibration_anomaly_score, electrical_charging_efficiency_score, vehicle_health_score]
    rul_features = [
        "thermal_stress_index",
        "brake_health_index",
        "mechanical_vibration_anomaly_score",
        "electrical_charging_efficiency_score",
        "vehicle_health_score"
    ]
    
    # Map user inputs to RUL features
    rul_input_data = {
        "thermal_stress_index": user_input["thermal_stress_index"],
        "brake_health_index": user_input["brake_health_index"],
        "mechanical_vibration_anomaly_score": user_input["vibration_score"],
        "electrical_charging_efficiency_score": user_input["electrical_efficiency"],
        "vehicle_health_score": 0.85 # Defaulting since it's not in the user's 7-feature list
    }
    rul_df = pd.DataFrame([rul_input_data], columns=rul_features)

    print("Performing predictions...")
    
    # RUL Predictions
    rul_engine = model_engine.predict(rul_df)[0]
    rul_brake = model_brake.predict(rul_df)[0]
    rul_battery = model_battery.predict(rul_df)[0]
    
    # Failure model expects: [engine_rul_pct, brake_rul_pct, battery_rul_pct, thermal_stress_index, mechanical_vibration_anomaly_score]
    failure_features = [
        "engine_rul_pct",
        "brake_rul_pct",
        "battery_rul_pct",
        "thermal_stress_index",
        "mechanical_vibration_anomaly_score"
    ]
    
    failure_input_data = {
        "engine_rul_pct": rul_engine,
        "brake_rul_pct": rul_brake,
        "battery_rul_pct": rul_battery,
        "thermal_stress_index": user_input["thermal_stress_index"],
        "mechanical_vibration_anomaly_score": user_input["vibration_score"]
    }
    failure_df = pd.DataFrame([failure_input_data], columns=failure_features)
    
    # Failure Prediction (Probability)
    failure_prob = model_failure.predict_proba(failure_df)[0][1] * 100 # In percentage

    # 4. Plotting
    print("Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use a colormap for dynamic colors
    colors = cm.viridis(np.linspace(0.3, 0.8, 3))
    
    # Plot 1: RUL Predictions
    components = ["Engine", "Brake", "Battery"]
    ruls = [rul_engine, rul_brake, rul_battery]
    axes[0].bar(components, ruls, color=colors)
    axes[0].set_title("Predicted Remaining Useful Life (RUL)")
    axes[0].set_ylabel("RUL (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Failure Probability
    axes[1].bar(["Failure Prob"], [failure_prob], color=cm.plasma(0.6))
    axes[1].set_title("Failure Probability (Next 7 Days)")
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 3: Feature Importance (Random Forest Failure Model)
    importances = model_failure.feature_importances_
    model_features = model_failure.feature_names_in_
    indices = np.argsort(importances)
    
    axes[2].barh(range(len(indices)), importances[indices], color=cm.viridis(0.5))
    axes[2].set_yticks(range(len(indices)))
    axes[2].set_yticklabels([model_features[i] for i in indices])
    axes[2].set_title("Feature Importance (Failure Model)")
    axes[2].set_xlabel("Importance Score")
    axes[2].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Final Summary Output
    print("\n--- Inference Results ---")
    print(f"Engine RUL:  {rul_engine:.2f}%")
    print(f"Brake RUL:   {rul_brake:.2f}%")
    print(f"Battery RUL: {rul_battery:.2f}%")
    print(f"Failure Prob (Next 7 days): {failure_prob:.2f}%")

if __name__ == "__main__":
    run_inference_and_plot()
