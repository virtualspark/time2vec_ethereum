# Ethereum Price Prediction using Time2Vec + LSTM

This project aims to develop an Ethereum price prediction machine learning model using Time2Vec. 
The stage of the project consists of the following:

1. Extracting Ethereum data from Yahoo Finance
2. Preprocessing data using MinMaxScaler
3. Define the Time2Vec model + LSTM. Generate sequence and labels to prepare data to feed in the model.
4. Use KerasGridSearch to find the best parameters (with the best score) to use for the Time2Vec + LSTM model
5. Apply the best parameters in the Time2Vec + LSTM model
6. Display the model loss with training and testing sets in plt graph
7. Display the actual and the predicted price in plt graph


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

#### Part 3 - Define the Time2Vec + LSTM model. Generate sequence and labels to prepare data to feed in the model, and perform train and test split. 

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
			
		
	##### To create X and Y for you #####
	def gen_sequence(id_df, seq_length, seq_cols):
			
		data_matrix = id_df[seq_cols].values
		num_elements = data_matrix.shape[0]

		for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
			yield data_matrix[start:stop, :]

	def gen_labels(id_df, seq_length, label):
			
		data_matrix = id_df[label].values
		num_elements = data_matrix.shape[0]
			
	return data_matrix[seq_length:num_elements, :]
			
		
		
	def T2V_NN(param, dim):
    
		inp = layers.Input(shape=(dim,1))
		x = T2V(param['t2v_dim'])(inp)
		x = LSTM(param['unit'], activation=param['act'])(x)
		x = Dense(1)(x)
			
		m = Model(inp, x)
		m.compile(loss='mse', optimizer='adam')
    
	return m
		
	##### PREPARE DATA TO FEED MODELS #####
	SEQ_LEN = 20 # pattern X is the size of Seq_len (e.g. use the first 20 days to predict 21st day)
	X, Y = [], []
	for sequence in gen_sequence(df, SEQ_LEN, ['Close']):
		X.append(sequence)
			
	for sequence in gen_labels(df, SEQ_LEN, ['Close']):
		Y.append(sequence)
			
	X = np.asarray(X)
	Y = np.asarray(Y)
		
	##### TRAIN TEST SPLIT #####

	train_dim = int(0.7*len(df))
	X_train, X_test = X[:train_dim], X[train_dim:]
	y_train, y_test = Y[:train_dim], Y[train_dim:]

	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)
		
Note that, for this part, the code is borrowed from this article: https://www.kaggle.com/danofer/time2vec-water-levels to create the Time2Vec + LSTM model.

#### Part 4 - Use KerasGridSearch to find the best parameters (with the best score) to use for the Time2Vec + LSTM model

	### DEFINE PARAMETER GRID FOR HYPERPARMETER OPTIMIZATION ###

	param_grid = {
		'unit': [64,32],
		't2v_dim': [128,64,16],
		'lr': [1e-2,1e-3], 
		'act': ['elu','relu'], 
		'epochs': 20,
		'batch_size': [128,512,1024]
	}
		
	hypermodel = lambda x: T2V_NN(param=x, dim=SEQ_LEN)

	kgs_t2v = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better=False, tuner_verbose=1)
	kgs_t2v.search(X_train, y_train, validation_split=0.2, shuffle=False)

Hyperparameters optimization is a big part of deep learning. Using KerasGridSearch enables you to find the best hyperparameters to tune your model.
You can refer to this article for more explanations regarding Grid Search: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
		
#### Part 5 - Apply the best parameters in the Time2Vec + LSTM model

	### Application of the parameters coming from the Keras Grid Search with the best score
	base_param = {
		'unit': 32,
		't2v_dim': 64,
		'lr': 1e-2, 
		'act': 'elu', 
		'epochs': 20,
		'batch_size': 1024
	}

	model = T2V_NN(param=base_param, dim=SEQ_LEN)
	history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, shuffle=False)
	model.evaluate(X_test, y_test)

After the best parameter had been generated in the Part 4, we use the best parameter in the model. 

#### Part 6 - Display the model loss with training and testing sets in plt graph

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

![image](https://user-images.githubusercontent.com/39383679/126915618-3a1a15e0-9901-466d-8cac-5128aa5a791e.png)

We are currently plotting the loss variables vs epoch to see if the loss had been reduced when we trained our model after several epochs.

#### Part 7 - Display the actual and the predicted price in plt graph

	y_hat = model.predict(X_test)

	# scale in a way that is easier to visualize in the graph (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
	# inverse_transform: Scale back the data to the original representation
	y_test_inverse = scaler.inverse_transform(y_test)
	y_hat_inverse = scaler.inverse_transform(y_hat)
	 
	plt.plot(y_test_inverse, label="Actual Price", color='green')
	plt.plot(y_hat_inverse, label="Predicted Price", color='red')
	 
	plt.title('Ethereum price prediction')
	plt.xlabel('Time [days]')
	plt.ylabel('Price')
	plt.legend(loc='best')
	 
	plt.show();

![image](https://user-images.githubusercontent.com/39383679/126915624-76d4590a-645a-4562-8410-f425c044fa58.png)

We are currently observing the graph to compare the actual price with the predicted price in the model. Note that we use the inverse_transform in order to scale data for better visualisation. 

## Author

Stanley Tran
