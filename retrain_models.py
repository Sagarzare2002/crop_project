import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_crop_model():
    print("Training Crop Model...")
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        # Preprocessing
        ms = MinMaxScaler()
        sc = StandardScaler()
        
        X_ms = ms.fit_transform(X)
        X_sc = sc.fit_transform(X_ms)
        
        # Training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_sc, y)
        
        # Saving
        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
        pickle.dump(sc, open('standscaler.pkl', 'wb'))
        
        print("Crop Model Trained and Saved Successfully!")
        
    except Exception as e:
        print(f"Error training Crop Model: {e}")

def train_fertilizer_model():
    print("Training Fertilizer Model...")
    try:
        df = pd.read_csv('f2.csv')
        
        # Standardize column names if needed
        # Expected: Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous, Fertilizer
        
        # Mappings (Normalized to lowercase for matching)
        soil_map = {
            'black': 0, 'clayey': 1, 'loamy': 2, 'red': 3, 'sandy': 4
        }
        
        crop_map = {
            'barley': 0, 'cotton': 1, 'ground nuts': 2, 'maize': 3, 'millets': 4,
            'oil seeds': 5, 'paddy': 6, 'pulses': 7, 'sugarcane': 8, 'tobacco': 9,
            'wheat': 10, 'coffee': 11, 'kidneybeans': 12, 'orange': 13,
            'pomegranate': 14, 'rice': 15, 'watermelon': 16
        }
        
        # Apply mappings
        df['Soil_Type'] = df['Soil_Type'].str.lower().map(soil_map)
        df['Crop_Type'] = df['Crop_Type'].str.lower().map(crop_map)
        
        # Check for unmapped values
        if df['Soil_Type'].isnull().any():
            print("Warning: Some Soil_Type values could not be mapped.")
            print(df[df['Soil_Type'].isnull()])
            
        if df['Crop_Type'].isnull().any():
            print("Warning: Some Crop_Type values could not be mapped.")
            print(df[df['Crop_Type'].isnull()])
            
        # Drop rows with missing values (if any mapping failed)
        df.dropna(subset=['Soil_Type', 'Crop_Type'], inplace=True)
        
        # Features and Target
        X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
        y = df['Fertilizer']
        
        # Encode Target
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Training
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        
        # Saving
        pickle.dump(model, open('classifier.pkl', 'wb'))
        pickle.dump(le, open('fertilizer.pkl', 'wb'))
        
        print("Fertilizer Model Trained and Saved Successfully!")
        
    except Exception as e:
        print(f"Error training Fertilizer Model: {e}")

if __name__ == "__main__":
    train_crop_model()
    train_fertilizer_model()
