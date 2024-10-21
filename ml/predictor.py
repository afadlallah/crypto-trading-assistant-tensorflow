from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import tensorflow as tf


class LSTMModel(Model):
    def __init__(self, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layers = [LSTM(hidden_dim, return_sequences=(i < num_layers - 1)) for i in range(num_layers)]
        self.fc = Dense(output_dim)

    def call(self, inputs):
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        return self.fc(x)


class PricePredictor:
    def __init__(self, lookback=60, prediction_horizon=1, epochs=100, batch_size=32, learning_rate=0.001, hidden_dim=100, num_layers=2, val_split=0.2, optimizer='adam', loss_fn='mse'):
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.val_split = val_split
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {self.device}")

    def prepare_data(self, data):
        close_data = data[['close']].values
        scaled_data = self.scaler.fit_transform(close_data)

        X = []
        y = []
        for i in range(self.lookback, len(scaled_data) - self.prediction_horizon + 1):
            X.append(scaled_data[i - self.lookback:i])
            y.append(scaled_data[i + self.prediction_horizon - 1, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        with tf.device(self.device):
            model = LSTMModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers, output_dim=1)
            model.build((None, input_shape[1], input_shape[2]))
        return model

    def train(self, data):
        X, y = self.prepare_data(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, shuffle=False)

        self.model = self.build_model(X_train.shape)

        # Select optimizer
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = SGD(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.get(self.optimizer)
        loss_fn = tf.keras.losses.get(self.loss_fn)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            with tf.device(self.device):
                # Train on batches
                train_loss = 0
                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i+self.batch_size]
                    batch_y = y_train[i:i+self.batch_size]
                    with tf.GradientTape() as tape:
                        predictions = self.model(batch_X, training=True)
                        predictions = tf.squeeze(predictions)
                        loss = loss_fn(batch_y, predictions)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    train_loss += loss.numpy()
                train_loss /= (len(X_train) // self.batch_size)

                # Evaluate on validation set
                val_predictions = self.model(X_val, training=False)
                val_predictions = tf.squeeze(val_predictions)
                val_loss = loss_fn(y_val, val_predictions).numpy()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        return history

    def predict(self, data):
        close_data = data[['close']].values
        scaled_data = self.scaler.transform(close_data)

        X = []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
        X = np.array(X)

        with tf.device(self.device):
            predictions = self.model.predict(X)

        return self.scaler.inverse_transform(predictions)

    def evaluate(self, data):
        X, y = self.prepare_data(data)
        with tf.device(self.device):
            y_pred = self.model.predict(X)
        mse = tf.keras.losses.mean_squared_error(y, y_pred)
        return tf.reduce_mean(mse).numpy()
