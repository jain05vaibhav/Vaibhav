# 🌌 Cloud AI Advisory Layer

> **Predictive Maintenance Intelligence for Advanced Vehicle Systems**
>
> This repository houses the **Cloud AI Advisory Layer**, a specialized intelligence module designed to analyze vehicle telemetry, predict Remaining Useful Life (RUL), and recommend proactive maintenance actions.

---

## 🚀 Quick Start

Follow these steps to get the system up and running in your environment.

### 1. Prerequisites
- **Python 3.14+** (Recommended)
- **Pip** (Internal package manager)
- **Active Internet Connection** (for Groq AI and API polling)

### 2. Environment Configuration
Create a `.env` file in the root directory and populate it with the following configuration:

```ini
# --- AI Configuration ---
GROQ_API_KEY=your_groq_api_key_here

# --- API Endpoints ---
SOURCE_GET_API_URL=https://fog-based-vehicle-monitoring.onrender.com/api/intelligence/data/unprocessed
DESTINATION_POST_API_URL=https://fog-based-vehicle-monitoring.onrender.com/api/insights/post_insights

# --- Polling Settings ---
POINTS_TO_FETCH=10
POLL_INTERVAL_SEC=10
VEHICLES_TO_MONITOR=VIT_CAR_001,VIT_CAR_002
```

### 3. Installation
Install the required Python libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Running the Service
Start the main polling service to begin processing vehicle data:

```bash
python main.py
```

---

## 🛠️ System Architecture

The Cloud AI Layer follows an advisory-only pattern, separating high-level intelligence from real-time vehicle control.

### Core Modules
- **`main.py`**: The orchestrator service that polls telemetry data, runs inference, and forwards insights.
- **`cloud_ai/`**: Contains the machine learning models and heuristic logic for:
  - **RUL Prediction**: Estimating life expectancy for Engines, Brakes, and Batteries.
  - **Failure Probability**: Calculating the risk of imminent breakdown.
  - **Explanation Engine**: Providing human-readable justifications for AI decisions.
  - **Recommendation Engine**: Suggesting maintenance actions (via Groq LLM or Rule-based fallback).

### Data Flow
1. **Fetch**: Retrieve unprocessed telemetry from the Fog/Edge layer.
2. **Analyze**: Blend current telemetry with historical trends using pre-trained ML models.
3. **Augment**: Generate RUL percentages, fault scores, and driver aggression indices.
4. **Deliver**: POST the enriched AI insights to the centralized dashboard.

---

## 📝 Key Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Your API key for Groq Cloud (Llama-based recommendations). | REQUIRED |
| `SOURCE_GET_API_URL` | Endpoint to fetch unprocessed vehicle data. | REQUIRED |
| `DESTINATION_POST_API_URL` | Endpoint to submit processed AI insights. | REQUIRED |
| `POINTS_TO_FETCH` | Number of historical data points to consider for trend analysis. | `10` |
| `POLL_INTERVAL_SEC` | Delay between polling cycles. | `10` |
| `VEHICLES_TO_MONITOR` | Comma-separated list of Vehicle IDs to track. | `VIT_CAR_001` |

---

> [!IMPORTANT]
> **Advisory Nature:** This system provides maintenance recommendations and predictive insights. It does **not** have actuation authority for safety-critical vehicle functions.

> [!TIP]
> Use Python **3.14** for optimal performance and compatibility with the latest asynchronous features used in the pipeline.

---
