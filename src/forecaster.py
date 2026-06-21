import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ImpactForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.encoders = {}
        self.features = ['latitude', 'longitude', 'event_type_encoded', 
                         'event_cause_encoded', 'priority_encoded', 
                         'start_hour', 'day_of_week', 'is_weekend']
        self.is_trained = False
        
    def train(self, df):
        """Trains the XGBoost model to predict impact duration."""
        print("Training ImpactForecaster...")
        cat_cols = ['event_type', 'event_cause', 'priority']
        df_encoded = df.copy()
        
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            le.fit(df_encoded[col].tolist() + ['unknown'])
            self.encoders[col] = le
            df_encoded[col + '_encoded'] = le.transform(df_encoded[col])
            
        X = df_encoded[self.features]
        y = np.log1p(df_encoded['impact_duration_mins'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        self.is_trained = True
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/forecaster_model.pkl')
        joblib.dump(self.encoders, 'models/forecaster_encoders.pkl')
        print("Model training complete and saved.")
        
    def predict(self, event_data):
        """Predicts impact duration and calculates congestion radius."""
        if not self.is_trained:
            self._load_model()
            
        df = pd.DataFrame([event_data])
        
        for col, le in self.encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val not in le.classes_:
                    val = 'unknown'
                df[col + '_encoded'] = le.transform([val])
            else:
                df[col + '_encoded'] = le.transform(['unknown'])
                
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0
                
        X = df[self.features]
        log_pred = self.model.predict(X)[0]
        duration_mins = np.expm1(log_pred)
        
        # Congestion radius scales with duration and priority
        base_radius = 500  
        priority_val = str(event_data.get('priority', '')).lower()
        if priority_val == 'high':
            base_radius = 800
        elif priority_val == 'low':
            base_radius = 300
            
        congestion_radius = min(base_radius + (duration_mins * 6), 4000) 
        
        return {
            'predicted_duration_mins': float(duration_mins),
            'congestion_radius_meters': float(congestion_radius)
        }
        
    def _load_model(self):
        try:
            self.model = joblib.load('models/forecaster_model.pkl')
            self.encoders = joblib.load('models/forecaster_encoders.pkl')
            self.is_trained = True
        except FileNotFoundError:
            raise RuntimeError("Model not trained yet. Call train() first.")
