# Ethereum Price Prediction using Time2Vec + LSTM

This project aims to develop an Ethereum price prediction machine learning model using Time2Vec. 
The stage of the project consists of the following:

1. Extracting Ethereum data from Yahoo Finance
2. Preprocessing data using MinMaxScaler
3. Define the Time2Vec + LSTM model (code borrowed from https://www.kaggle.com/danofer/time2vec-water-levels). 
4. Generate sequence and labels to prepare data to feed in the model
5. Use KerasGridSearch to find the best parameters (with the best score) to use for the Time2Vec + LSTM model
6. Apply the best parameters in the Time2Vec + LSTM model
7. Display the model loss with training and testing sets in plt graph
8. Display the actual and the predicted price in plt graph


## Getting Started

### Dependencies
1. Financial data parsing library: YFinance.
2. Grid Search Hyperparameters library: Keras-Hypertune.
3. Machine Learning Libraries: Tensorflow, Keras

### Cloning

To clone the git repository:

    git clone https://github.com/virtualspark/time2vec_ethereum

### Understanding of the program

#### Part 1 - Extracting data of Ethereum using Yahoo Finance

    df = pdr.get_data_yahoo('ETH-USD', '2016-01-01', '2021-07-19')

Note that we are currently using approximately 5 years of data for this project.

#### Part 2 - Preprocessing the data using MinMaxScaler

    scaler = MinMaxScaler()
    close_price = df.Close.values.reshape(-1, 1) # -1 in reshape function is used when you dont know or want to explicitly tell the dimension of that axis
    scaled_close = scaler.fit_transform(close_price) # This method performs fit and transform on the input data at a single time and converts the data points

#### Part 3 - Define the Time2Vec + LSTM model

    from tensorflow.keras import backend as K # Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the “backend engine” of Keras.
    from tensorflow.keras.layers import Layer

    class T2V(Layer):
    
        def __init__(self, output_dim=None, **kwargs):
            self.output_dim = output_dim
            super(T2V, self).__init__(**kwargs)

        def build(self, input_shape):

            self.W = self.add_weight(name='W',
                                    shape=(input_shape[-1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)

            self.P = self.add_weight(name='P',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)

            self.w = self.add_weight(name='w',
                                    shape=(input_shape[1], 1),
                                    initializer='uniform',
                                    trainable=True)

            self.p = self.add_weight(name='p',
                                    shape=(input_shape[1], 1),
                                    initializer='uniform',
                                    trainable=True)

            super(T2V, self).build(input_shape)

        def call(self, x):

            original = self.w * x + self.p #if i = 0
            sin_trans = K.sin(K.dot(x, self.W) + self.P) # Frequecy and phase shift of sine function, learnable parameters. if 1 <= i <= k

            return K.concatenate([sin_trans, original], -1)
