import pandas as pd
import jax.numpy as jnp
import jax
from jax import grad, jit
from jax import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time

def get_data():
    dataset_train = pd.read_csv('Data/train.csv')
    dataset_train = dataset_train.to_numpy()
    y_train = dataset_train[:, -1]
    X_train = dataset_train[:, :-1]


    dataset_test = pd.read_csv('Data/test.csv')
    dataset_test = dataset_test.to_numpy()
    y_test = dataset_test[:, -1]
    X_test = dataset_test[:, :-1]

    return (X_train, y_train), (X_test, y_test)



#Loading Data
(X_train, y_train), (X_test, y_test) = get_data()




# Initialize parameters
key = random.PRNGKey(0)
params = random.uniform(key, shape= (X_train.shape[1],))
velocity = jnp.zeros_like(params)  # Initialize velocity for momentum

# Model
@jit
def predict(params, X):
    return jnp.dot(X, params)

# Loss function (Mean Squared Error)
@jit
def mse_loss(params, X, y):
    preds = predict(params, X)
    return jnp.mean((preds - y) ** 2)


# SGD with Momentum update
@jit
def update(params, X, y, lr=0.01):
    grads = grad(mse_loss)(params, X, y)
    params = params - lr * grads

    return params


# Training loop
def train(params, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):

    training_times = []
    start_epoch = time.time()
    for epoch in range(epochs):
        params = update(params, X_train, y_train, lr)


        if epoch % 100 == 0:
            epoch_time = time.time() - start_epoch
            training_times.append(epoch_time)
            train_loss = mse_loss(params, X_train, y_train)
            test_loss = mse_loss(params, X_test, y_test)
            print(f'Epoch {epoch} | {str(epoch_time)[:5]} | Train Loss: {train_loss} | Test Loss: {test_loss}')
            start_epoch = time.time()

    return params, training_times



# Train the model
learning_rate = 0.00001
epochs = 20000
params, training_times = train(params, X_train, y_train, X_test, y_test,
                               epochs=epochs,
                               lr=learning_rate)

# Predictions
y_train_pred = np.array(predict(params, X_train))
y_test_pred = np.array(predict(params, X_test))

# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics and hyperparameteres
print(f'Learning Rate : {learning_rate} - Epochs : {epochs}')
print(f'Average training time per 100 epochs : {sum(training_times)/len(training_times)}')
print(f'Training Device : {jax.devices()}')
print(f'Final Train MSE: {train_mse}')
print(f'Final Test MSE: {test_mse}')

print(f'Final Train MAE: {train_mae}')
print(f'Final Test MAE: {test_mae}')

print(f'Final Train R2: {train_r2}')
print(f'Final Test R2: {test_r2}')
