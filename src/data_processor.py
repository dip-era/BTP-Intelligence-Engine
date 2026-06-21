import pandas as pd
import numpy as np
from datetime import datetime
import os

class TrafficDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Loads data from the given path."""
        print(f"Loading data from {self.data_path}...")
        self.raw_data = pd.read_csv(self.data_path)
        return self.raw_data
        
    def process(self):
        """
        Cleans and transforms the raw data for forecasting.
        Calculates impact duration and spatial/temporal features.
        """
        if self.raw_data is None:
            self.load_data()
            
        print("Processing data...")
        df = self.raw_data.copy()
        
        # Parse datetime columns
        date_cols = ['start_datetime', 'end_datetime', 'closed_datetime', 'resolved_datetime', 'modified_datetime']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                
        # Calculate event end time
        end_time_series = df['end_datetime']
        for col in ['resolved_datetime', 'closed_datetime', 'modified_datetime']:
            if col in df.columns:
                end_time_series = end_time_series.fillna(df[col])
                
        df['actual_end_datetime'] = end_time_series
        
        # Calculate duration in minutes
        df['impact_duration_mins'] = (df['actual_end_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
        
        # Filter negative or extremely long anomalies (e.g. > 10 days)
        df = df[(df['impact_duration_mins'] > 0) & (df['impact_duration_mins'] < 14400)]
        
        # Spatial features
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Filter valid Bengaluru coordinates approx
        df = df[(df['latitude'] > 12.5) & (df['latitude'] < 13.5) & 
                (df['longitude'] > 77.0) & (df['longitude'] < 78.0)]
                
        # Temporal features
        df['start_hour'] = df['start_datetime'].dt.hour
        df['day_of_week'] = df['start_datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Categorical variables
        df['event_type'] = df['event_type'].fillna('unknown').astype(str).str.lower()
        df['event_cause'] = df['event_cause'].fillna('unknown').astype(str).str.lower()
        df['priority'] = df['priority'].fillna('Medium')
        
        # Selecting relevant columns
        feature_cols = [
            'id', 'latitude', 'longitude', 'event_type', 'event_cause', 
            'priority', 'start_hour', 'day_of_week', 'is_weekend', 
            'impact_duration_mins'
        ]
        
        if 'description' in df.columns:
            feature_cols.append('description')
        if 'address' in df.columns:
            feature_cols.append('address')
            
        self.processed_data = df[feature_cols].copy()
        print(f"Data processed successfully. Shape: {self.processed_data.shape}")
        return self.processed_data
