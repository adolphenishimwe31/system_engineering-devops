Readme.md
"""
================================================================================
INTELLIGENT SOIL COMPACTION MONITORING SYSTEM
Real-time Degree of Compaction (%) Prediction using IoT and Machine Learning
================================================================================

ENGINEERING BASIS:
- AASHTO M 145 (Soil Classification)
- ASTM D698 (Standard Proctor)
- ASTM D1557 (Modified Proctor)
- ASTM D1556 (Sand Cone Method)
- ASTM D2922 (Nuclear Density Gauge)

SYSTEM ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────┐
│                        FIELD DEPLOYMENT                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ IoT Sensor Module (on Roller/Compactor)                      │  │
│  │  • Accelerometer (100Hz) - Vibration response               │  │
│  │  • Gyroscope (100Hz) - Angular velocity                     │  │
│  │  • Soil Moisture Sensor (1Hz) - Volumetric water content    │  │
│  │  • Temperature Sensor (1Hz) - Ambient & soil temp           │  │
│  │  • Load Cell (10Hz) - Contact force                         │  │
│  │  • GPS Module (1Hz) - Position tracking                     │  │
│  │  • Pass Counter - Roller pass number                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Edge Device (Raspberry Pi 4 / ESP32)                         │  │
│  │  • Real-time feature extraction                             │  │
│  │  • ML inference engine                                      │  │
│  │  • Online learning module                                   │  │
│  │  • MQTT client                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ MQTT/HTTP
┌─────────────────────────────────────────────────────────────────────┐
│                        CLOUD PLATFORM                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Data Pipeline                                                │  │
│  │  • Time-series database (InfluxDB)                          │  │
│  │  • Model retraining scheduler                               │  │
│  │  • Multi-site learning aggregation                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Dashboard & Alerts                                           │  │
│  │  • Real-time compaction map                                 │  │
│  │  • Pass/Fail notifications                                  │  │
│  │  • Historical reports                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

DEGREE OF COMPACTION CALCULATION:
    DC(%) = (γ_d_field / γ_d_max_proctor) × 100
    
    Where:
    - γ_d_field: Field dry density (pcf or kg/m³)
    - γ_d_max_proctor: Maximum dry density from Proctor test
    - Typical specifications: DC ≥ 95% for embankments, ≥ 98% for pavements

SYSTEM ADVANTAGES OVER TRADITIONAL METHODS:
1. Sand Cone Test (ASTM D1556):
   - Traditional: 15-30 min per test, destructive, sparse coverage
   - IoT System: Continuous, non-destructive, 100% coverage
   
2. Nuclear Density Gauge (ASTM D2922):
   - Traditional: Radiation safety concerns, spot measurements
   - IoT System: Safe, continuous monitoring
   
3. Core Cutter Method (ASTM D2937):
   - Traditional: Slow, requires lab analysis
   - IoT System: Instant results

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For edge deployment
try:
    import pickle
except ImportError:
    pass

# ============================================================================
# SECTION 1: SENSOR DATA SIMULATION AND PREPROCESSING
# ============================================================================

class SensorDataSimulator:
    """
    Simulates realistic sensor data from compaction equipment.
    
    Based on typical field conditions:
    - Soil types: Clay (CH, CL), Silt (ML, MH), Sand (SP, SW), Laterite
    - Moisture content affects compaction efficiency (AASHTO T99/T180)
    - Vibration frequency relates to compaction energy transfer
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.soil_types = ['Clay-CH', 'Clay-CL', 'Silt-ML', 'Sand-SP', 'Sand-SW', 'Laterite']
        
        # Maximum dry densities from Proctor tests (kg/m³)
        self.max_dry_densities = {
            'Clay-CH': 1650,  # High plasticity clay
            'Clay-CL': 1750,  # Low plasticity clay
            'Silt-ML': 1800,  # Low plasticity silt
            'Sand-SP': 1950,  # Poorly graded sand
            'Sand-SW': 2050,  # Well graded sand
            'Laterite': 1900  # Tropical residual soil
        }
        
        # Optimum moisture content (%)
        self.optimum_moisture = {
            'Clay-CH': 18,
            'Clay-CL': 15,
            'Silt-ML': 12,
            'Sand-SP': 8,
            'Sand-SW': 10,
            'Laterite': 14
        }
    
    def generate_training_data(self, n_samples=5000):
        """
        Generate labeled training data simulating lab + field calibration.
        
        This represents the initial phase where we have:
        1. Lab Proctor test results (γ_d_max)
        2. Field density tests (sand cone/nuclear gauge) for ground truth
        3. Simultaneous sensor readings during calibration
        """
        data = []
        
        for _ in range(n_samples):
            # Random soil type for this sample
            soil_type = np.random.choice(self.soil_types)
            max_dd = self.max_dry_densities[soil_type]
            opt_moisture = self.optimum_moisture[soil_type]
            
            # Number of compaction passes (1-12)
            n_passes = np.random.randint(1, 13)
            
            # Moisture content (% of dry weight)
            # Variation around optimum affects compaction efficiency
            moisture = opt_moisture + np.random.normal(0, 3)
            moisture = np.clip(moisture, 5, 25)
            
            # Distance from optimum moisture (critical parameter)
            moisture_deviation = abs(moisture - opt_moisture)
            
            # Roller speed (km/h) - slower is better for compaction
            speed = np.random.uniform(2, 8)
            
            # Contact force (kN) - relates to compaction energy
            contact_force = np.random.uniform(50, 150)
            
            # Temperature (°C) - affects soil behavior
            temperature = np.random.uniform(15, 40)
            
            # === ACCELEROMETER DATA (Vibration Response) ===
            # Higher vibration amplitude indicates less stiff soil (under-compacted)
            # Well-compacted soil shows higher frequency, lower amplitude
            base_vibration_amplitude = np.random.uniform(2, 8)  # m/s²
            vibration_freq = np.random.uniform(20, 45)  # Hz
            
            # Compaction affects vibration response
            compaction_factor = min(n_passes / 8, 1.0)
            vibration_amplitude = base_vibration_amplitude * (1 - 0.4 * compaction_factor)
            vibration_amplitude += np.random.normal(0, 0.5)
            
            # === GYROSCOPE DATA ===
            # Angular velocity variations (rad/s)
            gyro_x = np.random.normal(0, 0.3)
            gyro_y = np.random.normal(0, 0.3)
            gyro_z = np.random.normal(0, 0.5)
            
            # === FEATURE: RMS of accelerometer (common in vibration analysis) ===
            accel_rms = np.sqrt(vibration_amplitude**2 + np.random.uniform(0, 1))
            
            # === Calculate TRUE Degree of Compaction ===
            # This is our ground truth from field density tests
            
            # Base compaction from passes (logarithmic relationship)
            base_dc = 80 + 15 * np.log(n_passes + 1)
            
            # Moisture effect (parabolic - optimal at optimum moisture)
            moisture_effect = -0.8 * moisture_deviation
            
            # Speed penalty (faster = less effective)
            speed_penalty = -0.5 * (speed - 3)
            
            # Force benefit (more force = better compaction)
            force_benefit = 0.08 * (contact_force - 50)
            
            # Soil type base efficiency
            soil_efficiency = {
                'Clay-CH': -3,  # Harder to compact
                'Clay-CL': 0,
                'Silt-ML': 2,
                'Sand-SP': 4,   # Easier to compact
                'Sand-SW': 3,
                'Laterite': 1
            }
            
            degree_of_compaction = (base_dc + moisture_effect + speed_penalty + 
                                   force_benefit + soil_efficiency[soil_type])
            
            # Add realistic noise
            degree_of_compaction += np.random.normal(0, 1.5)
            
            # Physical constraints
            degree_of_compaction = np.clip(degree_of_compaction, 75, 102)
            
            # Calculate field dry density from DC
            field_dry_density = (degree_of_compaction / 100) * max_dd
            
            data.append({
                'soil_type': soil_type,
                'n_passes': n_passes,
                'moisture_content': moisture,
                'moisture_deviation': moisture_deviation,
                'roller_speed': speed,
                'contact_force': contact_force,
                'temperature': temperature,
                'vibration_amplitude': vibration_amplitude,
                'vibration_freq': vibration_freq,
                'accel_rms': accel_rms,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'max_dry_density': max_dd,
                'optimum_moisture': opt_moisture,
                'field_dry_density': field_dry_density,
                'degree_of_compaction': degree_of_compaction
            })
        
        return pd.DataFrame(data)


class SignalProcessor:
    """
    Real-time signal processing for sensor data.
    
    Implements noise filtering and feature extraction techniques
    commonly used in geotechnical instrumentation.
    """
    
    @staticmethod
    def butterworth_filter(data, cutoff_freq=10, sampling_rate=100, order=4):
        """
        Digital Butterworth low-pass filter for noise reduction.
        
        In production, use scipy.signal.butter. Here we provide the concept.
        High-frequency noise (>cutoff) is attenuated.
        """
        # Simplified simulation of filtered data
        noise_reduction = 0.7
        filtered = data + np.random.normal(0, np.std(data) * (1 - noise_reduction), 
                                          size=len(data))
        return filtered
    
    @staticmethod
    def extract_statistical_features(window_data):
        """
        Extract features from time-series window (e.g., 1-second windows).
        
        Common in vibration analysis for structural health monitoring.
        """
        features = {
            'mean': np.mean(window_data),
            'std': np.std(window_data),
            'rms': np.sqrt(np.mean(window_data**2)),
            'peak': np.max(np.abs(window_data)),
            'kurtosis': np.mean((window_data - np.mean(window_data))**4) / 
                       (np.std(window_data)**4) if np.std(window_data) > 0 else 0
        }
        return features
    
    @staticmethod
    def moving_average(data, window_size=10):
        """Simple moving average for smoothing."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Transform raw sensor data into meaningful features for ML model.
    
    Engineering principles:
    - Compaction energy correlates with density gain
    - Moisture-density relationship is parabolic (Proctor curve)
    - Vibration response indicates soil stiffness
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def engineer_features(self, df, fit=True):
        """
        Create derived features based on soil mechanics principles.
        """
        df_eng = df.copy()
        
        # === Compaction Energy Features ===
        # Energy per pass (simplified): E = Force × Distance
        # Higher energy → better compaction
        df_eng['energy_per_pass'] = df_eng['contact_force'] * (1 / df_eng['roller_speed'])
        
        # Cumulative compaction effort
        df_eng['cumulative_energy'] = df_eng['n_passes'] * df_eng['energy_per_pass']
        
        # === Moisture Relationship Features ===
        # Distance from optimum moisture (both sides of Proctor curve matter)
        df_eng['moisture_deviation_sq'] = df_eng['moisture_deviation'] ** 2
        
        # Relative moisture (normalized to optimum)
        df_eng['moisture_ratio'] = df_eng['moisture_content'] / df_eng['optimum_moisture']
        
        # === Vibration-based Stiffness Indicators ===
        # Stiffness proxy: lower amplitude + higher freq = stiffer soil
        df_eng['stiffness_indicator'] = (df_eng['vibration_freq'] / 
                                         (df_eng['vibration_amplitude'] + 0.1))
        
        # === Gyroscope Magnitude ===
        df_eng['gyro_magnitude'] = np.sqrt(df_eng['gyro_x']**2 + 
                                           df_eng['gyro_y']**2 + 
                                           df_eng['gyro_z']**2)
        
        # === Pass Efficiency ===
        # Diminishing returns after optimal passes
        df_eng['pass_efficiency'] = np.log(df_eng['n_passes'] + 1)
        
        # === Temperature-Moisture Interaction ===
        # Temperature affects moisture evaporation and soil behavior
        df_eng['temp_moisture_interaction'] = df_eng['temperature'] * df_eng['moisture_content']
        
        # One-hot encode soil type
        df_eng = pd.get_dummies(df_eng, columns=['soil_type'], prefix='soil')
        
        return df_eng
    
    def prepare_features(self, df_eng, target_col='degree_of_compaction', fit=True):
        """
        Select and scale features for model training.
        """
        # Feature columns (exclude target and intermediate columns)
        exclude_cols = [target_col, 'field_dry_density', 'max_dry_density', 
                       'optimum_moisture']
        feature_cols = [col for col in df_eng.columns if col not in exclude_cols]
        
        X = df_eng[feature_cols].values
        y = df_eng[target_col].values if target_col in df_eng.columns else None
        
        # Standardization (μ=0, σ=1) improves neural network convergence
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted before transform")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, feature_cols


# ============================================================================
# SECTION 3: MACHINE LEARNING MODELS
# ============================================================================

class CompactionPredictor:
    """
    Ensemble of ML models for degree of compaction prediction.
    
    Model selection rationale:
    1. Random Forest: Handles non-linear relationships, robust to outliers
    2. Gradient Boosting: High accuracy, captures complex interactions
    3. Neural Network: Flexible, can learn complex patterns
    """
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.trained = False
        
        # Initialize models
        if model_type in ['ensemble', 'random_forest']:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        if model_type in ['ensemble', 'gradient_boosting']:
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
        
        if model_type in ['ensemble', 'neural_network']:
            self.models['nn'] = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
    
    def train(self, df_train, target_col='degree_of_compaction'):
        """
        Train models on labeled data.
        
        Training Phase:
        - Use lab Proctor test results + field density measurements
        - Typical dataset: 500-5000 samples from calibration period
        """
        print("=" * 70)
        print("TRAINING PHASE: Initial Model Training")
        print("=" * 70)
        
        # Feature engineering
        df_eng = self.feature_engineer.engineer_features(df_train, fit=True)
        X, y, feature_cols = self.feature_engineer.prepare_features(
            df_eng, target_col, fit=True
        )
        self.feature_cols = feature_cols
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            print(f"\n--- Training {name.upper()} ---")
            model.fit(X_train, y_train)
            
            # Validation predictions
            y_pred = model.predict(X_val)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"RMSE: {rmse:.3f}%")
            print(f"MAE:  {mae:.3f}%")
            print(f"R²:   {r2:.4f}")
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-5:]
                print("\nTop 5 Important Features:")
                for idx in reversed(top_idx):
                    print(f"  {feature_cols[idx]}: {importances[idx]:.4f}")
        
        self.trained = True
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        return results
    
    def predict(self, df_new):
        """
        Real-time prediction phase.
        
        Input: Raw sensor data from current measurement
        Output: Degree of compaction (%)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Feature engineering (using fitted scaler)
        df_eng = self.feature_engineer.engineer_features(df_new, fit=False)
        X, _, _ = self.feature_engineer.prepare_features(df_eng, 
                                                         target_col=None, 
                                                         fit=False)
        
        # Ensemble prediction (average of all models)
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average ensemble prediction
        final_prediction = np.mean(predictions, axis=0)
        
        # Also calculate uncertainty (std of predictions)
        prediction_std = np.std(predictions, axis=0)
        
        return final_prediction, prediction_std
    
    def save_model(self, filepath='compaction_model.pkl'):
        """Save trained model for edge deployment."""
        model_package = {
            'models': self.models,
            'feature_engineer': self.feature_engineer,
            'feature_cols': self.feature_cols,
            'trained': self.trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='compaction_model.pkl'):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        predictor = cls()
        predictor.models = model_package['models']
        predictor.feature_engineer = model_package['feature_engineer']
        predictor.feature_cols = model_package['feature_cols']
        predictor.trained = model_package['trained']
        
        return predictor


# ============================================================================
# SECTION 4: ONLINE LEARNING (INCREMENTAL LEARNING)
# ============================================================================

class OnlineLearningModule:
    """
    Continuous learning system that improves model during field operation.
    
    Strategy:
    1. Collect sensor data + occasional field verification tests
    2. Update model incrementally with new verified samples
    3. Avoid catastrophic forgetting using experience replay
    
    Based on:
    - SGDRegressor for linear models (incremental updates)
    - Mini-batch learning for neural networks
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        
        # Online learner (SGDRegressor supports partial_fit)
        self.online_model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42
        )
        
        # Experience replay buffer (stores recent verified samples)
        self.replay_buffer = []
        self.buffer_size = 1000
        
        self.online_trained = False
    
    def initialize_online_learning(self, X_init, y_init):
        """Initialize online model with base training data."""
        self.online_model.fit(X_init, y_init)
        self.online_trained = True
        print("Online learning module initialized")
    
    def partial_update(self, X_new, y_new):
        """
        Incremental update with new verified field data.
        
        Called when:
        - Spot check with sand cone/nuclear gauge confirms DC
        - High-confidence predictions are verified
        
        Frequency: ~5-10 samples per day during construction
        """
        if not self.online_trained:
            raise ValueError("Online model must be initialized first")
        
        # Add to replay buffer
        for x, y in zip(X_new, y_new):
            self.replay_buffer.append((x, y))
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)  # Remove oldest
        
        # Update with new data
        self.online_model.partial_fit(X_new, y_new)
        
        # Periodic replay to prevent forgetting (every 10 updates)
        if len(self.replay_buffer) > 50 and np.random.rand() < 0.1:
            sample_size = min(50, len(self.replay_buffer))
            replay_samples = np.random.choice(len(self.replay_buffer), 
                                             sample_size, replace=False)
            X_replay = np.array([self.replay_buffer[i][0] for i in replay_samples])
            y_replay = np.array([self.replay_buffer[i][1] for i in replay_samples])
            self.online_model.partial_fit(X_replay, y_replay)
    
    def predict_with_adaptation(self, X):
        """Prediction using adapted model."""
        if self.online_trained:
            return self.online_model.predict(X)
        else:
            return self.base_model.predict(X)


# ============================================================================
# SECTION 5: EDGE DEPLOYMENT
# ============================================================================

class EdgeInferenceEngine:
    """
    Lightweight inference for edge devices (Raspberry Pi, ESP32).
    
    Optimization strategies:
    1. Model quantization (reduce float32 → int8)
    2. Feature caching (reuse computed features)
    3. Batch processing for efficiency
    
    Typical edge specs:
    - Raspberry Pi 4: 4GB RAM, ARM Cortex-A72
    - ESP32: 520KB RAM, dual-core 240MHz
    
    For ESP32: Use TensorFlow Lite for Microcontrollers
    For RPi: Full scikit-learn model works fine
    """
    
    def __init__(self, model, target_platform='raspberry_pi'):
        self.model = model
        self.platform = target_platform
        self.feature_cache = {}
    
    def inference(self, sensor_reading):
        """
        Real-time inference on edge device.
        
        Input: Dictionary with current sensor values
        Output: (DC%, uncertainty, pass/fail)
        """
        # Convert to DataFrame
        df = pd.DataFrame([sensor_reading])
        
        # Predict
        dc_pred, uncertainty = self.model.predict(df)
        
        # Pass/Fail logic (specification: ≥95% for embankment, ≥98% for pavement)
        specification = sensor_reading.get('specification', 95.0)
        pass_fail = "PASS" if dc_pred[0] >= specification else "FAIL"
        
        result = {
            'degree_of_compaction': float(dc_pred[0]),
            'uncertainty': float(uncertainty[0]),
            'status': pass_fail,
            'specification': specification,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def estimate_inference_time(self):
        """
        Estimate inference latency.
        
        Typical values:
        - Raspberry Pi 4: 5-10ms
        - ESP32 (TFLite): 50-100ms
        """
        if self.platform == 'raspberry_pi':
            return "~10ms"
        elif self.platform == 'esp32':
            return "~80ms"
        else:
            return "Unknown"


# ============================================================================
# SECTION 6: COMMUNICATION & DASHBOARD
# ============================================================================

class MQTTCommunicator:
    """
    MQTT communication for IoT data transmission.
    
    Protocol: MQTT (Message Queue Telemetry Transport)
    - Lightweight publish/subscribe messaging
    - Perfect for constrained devices
    - QoS levels for reliability
    
    Topics:
    - sensors/{device_id}/data  → Raw sensor data
    - predictions/{device_id}/dc → Compaction predictions
    - alerts/{device_id}/fail → Alert when DC < spec
    """
    
    def __init__(self, broker_address="mqtt.example.com", port=1883):
        self.broker = broker_address
        self.port = port
        print(f"MQTT Communicator initialized: {broker_address}:{port}")
    
    def publish_prediction(self, device_id, prediction_result):
        """
        Publish prediction to cloud dashboard.
        
        In production, use paho-mqtt library:
        ```python
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect(self.broker, self.port)
        client.publish(f"predictions/{device_id}/dc", json.dumps(prediction_result))
        ```
        """
        topic = f"predictions/{device_id}/dc"
        message = json.dumps(prediction_result)
        print(f"[MQTT] Publishing to {topic}: {message}")
    
    def publish_alert(self, device_id, alert_data):
        """Send alert when compaction fails specification."""
        topic = f"alerts/{device_id}/fail"
        message = json.dumps(alert_data)
        print(f"[ALERT] {topic}: {message}")


class DashboardLogic:
    """
    Backend logic for web/mobile dashboard.
    
    Features:
    - Real-time compaction heatmap (GPS + DC%)
    - Historical trends
    - Pass/Fail notifications
    - Export reports (PDF/Excel)
    
    Tech stack suggestion:
    - Frontend: React.js with Leaflet.js for maps
    - Backend: FastAPI or Flask
    - Database: InfluxDB (time
