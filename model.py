import preprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

def train_model(X, epochs=50, batch_size=32):
   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(X.shape[1], activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, X, epochs=epochs, batch_size=batch_size)

   
    y_pred = model.predict(X)
    mse = mean_squared_error(X, y_pred)
    print('MSE:', mse)

    return model
