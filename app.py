from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Load dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'BMTC_Passenger_Traffic_Dataset_Full.csv')
df_buses = pd.read_csv(DATASET_PATH)

# Constants
TRAVEL_TIME_BASE = 20    # Base travel time in minutes
TRAVEL_TIME_PER_BUS = 2  # Additional minutes per bus (you can adjust this as needed)

# -------------------- ML MODEL (in-memory) --------------------
_model = None
_feature_columns = ['route no', 'time']
_day_columns = [f'day{i}' for i in range(1, 32)]

# Precomputed historical averages: {(route, hour) -> avg}
_hist_avg_map = {}

# Route-level multipliers to force route-specific variation if underlying patterns are similar
_route_multiplier = {}
# Small per-route bias added post-scale to nudge ceil differences
_route_bias = {}


def _prepare_training_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()
    df['avg_buses'] = df[_day_columns].mean(axis=1)
    df = df[_feature_columns + ['avg_buses']].dropna(subset=['avg_buses'])
    df['time'] = df['time'].astype(float)
    df['route no'] = df['route no'].astype(str).str.strip()
    return df


def _build_hist_avg_map(source_df: pd.DataFrame) -> dict:
    grouped = source_df.copy()
    grouped['route no'] = grouped['route no'].astype(str).str.strip()
    grouped['time'] = grouped['time'].astype(float)
    grouped['avg_buses'] = grouped[_day_columns].mean(axis=1)
    result = {}
    for _, row in grouped.iterrows():
        key = (row['route no'], float(row['time']))
        result[key] = float(row['avg_buses']) if not np.isnan(row['avg_buses']) else 0.0
    return result


def _init_route_multipliers(source_df: pd.DataFrame) -> dict:
    # Assign a stable, route-specific multiplier in [0.70, 1.50) using a hash of the route string.
    import hashlib
    tmp = source_df.copy()
    tmp['route no'] = tmp['route no'].astype(str).str.strip()
    unique_routes = sorted(tmp['route no'].unique())
    mults = {}
    biases = {}
    for route in unique_routes:
        h = hashlib.md5(route.encode('utf-8')).hexdigest()
        val = int(h[:8], 16)
        u = (val % 10000) / 10000.0  # [0,1)
        m = 0.70 + 0.80 * u           # [0.70, 1.50)
        # Derive a tiny bias in [0.00, 0.40) to influence ceiling step
        v = int(h[8:16], 16)
        b = (v % 400) / 1000.0        # [0.000, 0.399]
        mults[route] = float(m)
        biases[route] = float(b)
    # Store biases globally as a side effect
    global _route_bias
    _route_bias = biases
    return mults


def train_model():
    global _model
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        train_df = _prepare_training_dataframe(df_buses)

        numeric_features = ['time']
        categorical_features = ['route no']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ]
        )

        rf = RandomForestRegressor(n_estimators=200, random_state=42)

        _model = Pipeline(steps=[('prep', preprocessor), ('model', rf)])
        _model.fit(train_df[_feature_columns], train_df['avg_buses'])
        return True, None
    except Exception as e:
        _model = None
        return False, str(e)


# Train on startup and build maps
_ok, _err = train_model()
if not _ok:
    print(f"[WARN] Model training failed on startup: {_err}")
_hist_avg_map = _build_hist_avg_map(df_buses)
_route_multiplier = _init_route_multipliers(df_buses)


def get_current_half_hour_slot():
    now = datetime.now()
    minute = 0 if now.minute < 30 else 30
    return now.replace(minute=minute, second=0, microsecond=0)


def _predict_avg_buses(route_no: str, hour: float) -> float:
    if _model is None:
        return np.nan
    try:
        X = pd.DataFrame([{ 'route no': str(route_no).strip(), 'time': float(hour) }])
        y_pred = _model.predict(X)[0]
        if np.isnan(y_pred):
            return np.nan
        return max(0.0, float(y_pred))
    except Exception:
        return np.nan


def _hist_lookup(route_no: str, hour: float) -> float:
    return float(_hist_avg_map.get((str(route_no).strip(), float(hour)), np.nan))


def _apply_route_multiplier(route_no: str, value: float) -> float:
    route = str(route_no).strip()
    m = float(_route_multiplier.get(route, 1.0))
    b = float(_route_bias.get(route, 0.0))
    return max(0.0, value * m + b)


def estimate_buses(route_no, hour):
    try:
        hour = float(hour)
        route_no = str(route_no).strip()

        ml_pred = _predict_avg_buses(route_no, hour)
        hist_avg = _hist_lookup(route_no, hour)

        values = []
        weights = []
        if not np.isnan(ml_pred):
            values.append(ml_pred)
            weights.append(0.7)
        if not np.isnan(hist_avg):
            values.append(hist_avg)
            weights.append(0.3)

        if values:
            avg_buses = float(np.average(values, weights=weights))
        else:
            entries = df_buses[(df_buses['route no'] == route_no) & (df_buses['time'] == hour)]
            if entries.empty:
                return {'current_est': 'N/A', 'next_est': 'N/A', 'travel_time': 'N/A'}
            avg_buses = entries[_day_columns].mean().mean()

        # Apply route multiplier to introduce route-specific variation
        avg_buses = _apply_route_multiplier(route_no, avg_buses)

        if np.isnan(avg_buses):
            avg_buses = 0

        buses_required = int(np.ceil(avg_buses))
        travel_time = int(TRAVEL_TIME_BASE + buses_required * TRAVEL_TIME_PER_BUS)

        next_hour = float((hour + 1) % 24)
        ml_next = _predict_avg_buses(route_no, next_hour)
        hist_next = _hist_lookup(route_no, next_hour)
        values = []
        weights = []
        if not np.isnan(ml_next):
            values.append(ml_next)
            weights.append(0.7)
        if not np.isnan(hist_next):
            values.append(hist_next)
            weights.append(0.3)
        if values:
            next_avg_buses = float(np.average(values, weights=weights))
        else:
            next_entries = df_buses[(df_buses['route no'] == route_no) & (df_buses['time'] == next_hour)]
            next_avg_buses = next_entries[_day_columns].mean().mean() if not next_entries.empty else avg_buses

        next_avg_buses = _apply_route_multiplier(route_no, next_avg_buses)

        if np.isnan(next_avg_buses):
            next_avg_buses = avg_buses
        next_buses_required = int(np.ceil(next_avg_buses))

        return {
            'current_est': buses_required,
            'next_est': next_buses_required,
            'travel_time': f'{travel_time} minutes'
        }
    except Exception as e:
        return {'current_est': 'Error', 'next_est': 'Error', 'travel_time': str(e)}


@app.route('/api/estimate', methods=['POST'])
def get_estimate():
    try:
        data = request.get_json()
        route_no = data.get('routeCode')
        time_str = data.get('time')  # Format: HH:MM

        hour = int(time_str.split(':')[0])
        result = estimate_buses(route_no, hour)

        route_info = df_buses[df_buses['route no'] == route_no][['source', 'destination']].iloc[0].to_dict()
        result.update(route_info)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    try:
        data = request.get_json()
        route_no = str(data.get('routeCode')).strip()

        forecast_data = []
        for hour in range(24):
            hist_avg = _hist_lookup(route_no, float(hour))
            if np.isnan(hist_avg):
                est = estimate_buses(route_no, hour)
                buses_required = est['current_est'] if isinstance(est['current_est'], int) else 0
            else:
                buses_required = int(np.ceil(_apply_route_multiplier(route_no, hist_avg)))
            travel_time = int(TRAVEL_TIME_BASE + int(buses_required) * TRAVEL_TIME_PER_BUS)

            forecast_data.append({
                'hour': f'{hour:02d}:00',
                'buses_required': buses_required,
                'travel_time': f'{travel_time} minutes'
            })

        route_info = df_buses[df_buses['route no'] == route_no][['source', 'destination']].iloc[0].to_dict()

        return jsonify({
            'forecast': forecast_data,
            'route_info': route_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/route-multipliers', methods=['GET', 'POST'])
def route_multipliers():
    global _route_multiplier, _route_bias
    if request.method == 'GET':
        return jsonify({'multipliers': _route_multiplier, 'bias': _route_bias})
    try:
        payload = request.get_json() or {}
        mults = payload.get('multipliers') or {}
        bias = payload.get('bias') or {}
        for k, v in mults.items():
            if not isinstance(v, (int, float)):
                continue
            _route_multiplier[str(k).strip()] = float(max(0.2, min(3.0, v)))
        for k, v in bias.items():
            if not isinstance(v, (int, float)):
                continue
            _route_bias[str(k).strip()] = float(max(0.0, min(1.0, v)))
        return jsonify({'status': 'ok', 'multipliers': _route_multiplier, 'bias': _route_bias})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/train', methods=['POST'])
def trigger_train():
    ok, err = train_model()
    if ok:
        # Rebuild historical map in case the CSV changed between runs
        global _hist_avg_map
        _hist_avg_map = _build_hist_avg_map(df_buses)
        global _route_multiplier
        _route_multiplier = _init_route_multipliers(df_buses)
        return jsonify({'status': 'trained'})
    return jsonify({'status': 'failed', 'error': err}), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def index_page():
    return render_template('index.html')


@app.route('/input.html')
def input_page():
    return render_template('input.html')


@app.route('/output.html')
def output_page():
    return render_template('output.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
