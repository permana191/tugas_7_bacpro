import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Data
df = pd.read_csv('noise_level_data.csv')
data_raw = df.iloc[:, 0].values[:10000].astype('float32') 

# 2. Windowing
def create_sequences(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(data_raw)

# 3. Scaling (GUNAKAN DUA SCALER BERBEDA)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X) # Untuk 5 fitur input
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)) # Untuk 1 fitur output

# 4. Model
model = Sequential([
    Input(shape=(5,)), 
    Dense(16, activation='relu'), 
    Dense(8, activation='relu'),    
    Dense(1, activation='linear')   
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_scaled, y_scaled, epochs=50, validation_split=0.2, callbacks=[early_stop])

# 6. SIMPAN KEDUA SCALER
model.save('model_jst.h5')
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Training Selesai! File model_jst.h5, scaler_X.pkl, dan scaler_y.pkl siap digunakan.")