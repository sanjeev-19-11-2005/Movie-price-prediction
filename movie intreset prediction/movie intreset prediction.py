import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv("/content/Movie Interests.csv")
features = ds[['Age', 'Gender']].values
target = ds['Interest'].values

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_max = np.max([len(interest) for interest in target]) # Find the maximum length of interest strings
y_train = np.array([len(interest) / y_max for interest in y_train])  # Normalize interest lengths
y_test = np.array([len(interest) / y_max for interest in y_test])   # Normalize interest lengths


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.fit(x_train, y_train, epochs=100, verbose=1)


loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}, Test MAE: {mae:.4f}")


input_age = 25
input_gender = 1
input_arr = np.array([[input_age, input_gender]])
input_arr = scaler.transform(input_arr)
prediction = model.predict(input_arr)
predicted_interest = prediction[0][0] * y_max
print(f"Predicted Movie Interest: {predicted_interest:.2f}")
