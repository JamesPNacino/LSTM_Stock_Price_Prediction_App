# app.py, this is the main file
from myproject import app, db #import app from the __init__.py file
from flask import render_template, render_template, redirect, url_for, make_response, session, jsonify, request
from flask_sqlalchemy import SQLAlchemy
#from flask_socketio import SocketIO, emit
from myproject import db
from myproject.models import neural_network #need this to create tables
from myproject.view_networks.forms import RunTickerModel_Form
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler #Used to normalize data
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
import re
import sqlite3
import datetime
import pickle
import gc



#Socket listens for real-time updates from the server and allows to emit messages to the user interface
#socketio = SocketIO(app)



@app.route('/', methods=['GET', 'POST'])
def INDEX():

	#Query DB for info on previously trained models
	model_table_df = pd.read_sql_query(f"SELECT * FROM neural_network", db.get_engine())
	model_table_df.columns = ['id', 'ticker', 'pattern', 'days_out_pmtr', 'pct_incrse_pmtr', 'total_observations_train_val', 'accuracy_test', 'majority_class_test_pct', 'start_date', 'end_date']
	model_table_df_copy = model_table_df #will use this table later if form is pressed to get ID field for trained_model table

	#create html table to display the previously trained models; used for html jinja
	model_table_df = model_table_df.to_html(classes='table table-striped', index=False)


	#query database for network information
	form = RunTickerModel_Form()

 	# Check if the request is an AJAX request
	#if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':  # This checks if the request is an AJAX request
     
	if form.validate_on_submit():
	    # Collect data 
		ticker_query = request.form['ticker_query'].upper()
		pattern = request.form['pattern']
		days_out = request.form['days_out']
		pct_increase = request.form['pct_increase']
		data_end_date = request.form['data_end_date']

		#Query for if there is a model that was trained on the same parameters; if so, we can skip training and continue to just load previously trained model's weights that were trained on same parameters
		previously_trained_df = pd.read_sql_query(f"SELECT * FROM neural_network WHERE ticker='{ticker_query}' AND pattern='{pattern}' AND days_out_parameter='{days_out}' AND percent_increase_parameter='{pct_increase}'AND data_end_date='{data_end_date}'", db.get_engine())
		

		#If a previously trained model has been found, show on the terminal, the following message
		if (len(previously_trained_df) > 0):
			neural_network_id = previously_trained_df["id"].iloc[0]
			previously_trained_message = f"Previous model was found using these parameters: {previously_trained_df}"
			print(previously_trained_message)
		else:
			#if there is not previously trained model found, set it to 'None'
			previously_trained_message = None
			neural_network_id = str(len(model_table_df_copy) + 1)


		ticker = ticker_query #stock ticker
		#ticker_symbol = ticker #used for the training step in a later section as need to save ticker data
		ticker = yf.Ticker(ticker)

		# Define the custom date range (start and end dates in 'YYYY-MM-DD' format)
		start_date = "2000-01-01" #Fixed start date at "2000-01-01"
		end_date = data_end_date

		# Fetch the historical data for the defined date range
		finance_df = ticker.history(start=start_date, end=end_date)

		#Remove the following columns - the only columns that should remain from the original data pull are ['Open','High','Low','Close','Volume']
		if 'Dividends' in finance_df.columns:
			finance_df = finance_df.drop('Dividends', axis=1)
		if 'Stock Splits' in finance_df.columns:
			finance_df = finance_df.drop('Stock Splits', axis=1)
		if 'Capital Gains' in finance_df.columns:
			finance_df = finance_df.drop('Capital Gains', axis=1)

		#creates index column
		finance_df['Row_index'] = range(1, len(finance_df) + 1) 
	
		#Creates a conditional statement based on conditions, and applies the 'choices' label.
		#This will result in a new column 'Candle', which describes if the daily candle was 'Bullish',
		#Bearish, or Neutral
		conditions = [finance_df['Close'] > finance_df['Open'], finance_df['Close'] == finance_df['Open'], finance_df['Open'] > finance_df['Close']]
		choices = ['Bullish', 'Neutral', 'Bearish']
		finance_df['Candle'] = np.select(conditions, choices, default='Unknown')

		#Create column 'Body_length' which calculates the body length of the candle. 
		finance_df['Body_length'] = abs(finance_df['Close'] - finance_df['Open']) #get the absolute number

		#Create column 'Lower_shadow_length' which calculates the distance between the low and the open for a bullish candle, or the low and the close for a bearish candle.
		conditions = [
    		finance_df['Candle'] == 'Bullish',
    		finance_df['Candle'] == 'Neutral',
    		finance_df['Candle'] == 'Bearish'
		]
		choices = [finance_df['Open'] - finance_df['Low'], 
           		finance_df['Open'] - finance_df['Low'], 
           		finance_df['Close'] - finance_df['Low']]
		finance_df['Lower_shadow_length'] = np.select(conditions, choices, default=0.0)

		#Create column 'Upper_shadow_length' which calculates the distance between the high and the close for a bullish candle, or the high and the open for a bearish candle.
		conditions = [
    		finance_df['Candle'] == 'Bullish',
    		finance_df['Candle'] == 'Neutral',
    		finance_df['Candle'] == 'Bearish'
		]
		choices = [finance_df['High'] - finance_df['Close'], 
           		finance_df['High'] - finance_df['Close'], 
           		finance_df['High'] - finance_df['Open']]
		finance_df['Upper_shadow_length'] = np.select(conditions, choices, default=0.0)

		#Create column 'Total_candle_length' which calculates the distance between low and high prices
		finance_df['Total_candle_length'] = finance_df['High'] - finance_df['Low']

		hammer = [] #initialize empty list
		for index, row in finance_df.iterrows():
			start_index = row['Row_index'] - 5 #Get the starting index of the fifth previous candle
			end_index = row['Row_index'] - 1 #subtacting 1 because only getting previous 5 candlestick closing prices, not current candle's close
			#for example, if row['Row_index'] = 6; start_index = 6 - 5 = 1; end_index = 6 - 1 = 5 --- getting rows 1 (start_index) through 5 (end_index) which 
			#is the previous five candles data since our current candle is the sixth candle
			
			
			if (start_index < 1): #since we are subtracting to get starting index, it will be negative numbers at first; skip these
				hammer.append("No")
				continue
				
			temp_df = finance_df[(finance_df['Row_index'] >= start_index) & (finance_df['Row_index'] <= end_index)]
			closing_prices = temp_df['Close'].values

			#Fit a regression line (line of best fit) through the closing prices; negative slope represent current downtrend
			x = np.arange(len(closing_prices))
			slope, intercept = np.polyfit(x, closing_prices, 1) 

			#if slope <= 0, then in a current downtrend the previous 5 days in terms of closing prices
			#to clarify, although slope may be zero which means no change, I will still be counting this as a downtrend
			if (slope <= 0):
				if ((row['Lower_shadow_length']) >= (row['Body_length'] * 2)): #lower shadow must be at least twice as long as body length
					if ((row['Upper_shadow_length']) < (row['Total_candle_length'] * 0.10)): #Upper shadow is less than 10% of the total candle length
						if ((row['Body_length']) < (row['Total_candle_length'] * 0.30)): #body is less than 30% of the total candlestick length
							hammer.append("Yes")
						else:
							hammer.append("No")
					else:
						hammer.append("No")
				else:
					hammer.append("No")
					
			elif (slope > 0):
				hammer.append("No")

		#Create new column
		finance_df['Hammer_pattern'] = hammer

		# Get the indices where 'Hammer_pattern' is 'Yes'
		yes_indices = finance_df[finance_df['Hammer_pattern'] == 'Yes'].index

		# Calculate how many "Yes" values should be changed to "Yes_test"
		num_yes_to_change = int(len(yes_indices) * 0.10)

		# Randomly sample indices from the "Yes" values
		#set seed for reproducibility
		np.random.seed(6) 
		indices_to_change = np.random.choice(yes_indices, num_yes_to_change, replace=False)

		# Change the "Yes" values at the sampled indices to "Yes_test"; only 10% of the "Yes" values will be changed to "Yes_test"
		finance_df.loc[indices_to_change, 'Hammer_pattern'] = 'Yes_test'


		inverted_hammer = [] #initialize empty list
		for index, row in finance_df.iterrows():
			start_index = row['Row_index'] - 5 #Get the starting index of the fifth previous candle
			end_index = row['Row_index'] - 1 #subtacting 1 because only getting previous 5 candlestick closing prices, not current candle's close
			
			if (start_index < 1): #since we are subtracting to get starting index, it will be a negative number; skip these
				inverted_hammer.append("No")
				continue
				
			
			temp_df = finance_df[(finance_df['Row_index'] >= start_index) & (finance_df['Row_index'] <= end_index)]
			closing_prices = temp_df['Close'].values

			#Fit a regression line (line of best fit) through the closing prices; negative slope represents current downtrend
			x = np.arange(len(closing_prices))
			slope, intercept = np.polyfit(x, closing_prices, 1) 

			#if slope <= 0, then in a current downtrend the previous 5 days in terms of closing prices
			#to clarify, although slope may be zero which means no change, I will still be counting this as a downtrend
			if (slope <= 0):
				if ((row['Upper_shadow_length']) >= (row['Body_length'] * 2.0)): #upper shadow is at least twice the length of the candle body
					if ((row['Lower_shadow_length']) < (row['Total_candle_length'] * 0.10)): #lower shadow is less than 20% of the total candle length
						if ((row['Body_length']) < (row['Total_candle_length'] * 0.30)): #body is less than 30% of the total candlestick length
							inverted_hammer.append("Yes")
						else:
							inverted_hammer.append("No")
					else:
						inverted_hammer.append("No")
				else:
					inverted_hammer.append("No")
						
			elif (slope > 0):
				inverted_hammer.append("No")

		#Create new column
		finance_df['InvertedHammer_pattern'] = inverted_hammer

		# Get the indices where 'Hammer_pattern' is 'Yes'
		yes_indices = finance_df[finance_df['InvertedHammer_pattern'] == 'Yes'].index

		# Calculate how many "Yes" values should be changed to "Yes_test"
		num_yes_to_change = int(len(yes_indices) * 0.10)

		# Randomly sample indices from the "Yes" values
		#set seed for reproducibility
		np.random.seed(6) 
		indices_to_change = np.random.choice(yes_indices, num_yes_to_change, replace=False)

		# Change the "Yes" values at the sampled indices to "Yes_test"; only 10% of the "Yes" values will be changed to "Yes_test"
		finance_df.loc[indices_to_change, 'InvertedHammer_pattern'] = 'Yes_test'


		###Calculate MACD and Signal Line
		# Calculate the 12-day EMA
		finance_df['EMA12'] = finance_df['Close'].ewm(span=12, adjust=False).mean()

		# Calculate the 26-day EMA
		finance_df['EMA26'] = finance_df['Close'].ewm(span=26, adjust=False).mean()

		# Calculate the MACD (12-day EMA - 26-day EMA)
		finance_df['MACD'] = finance_df['EMA12'] - finance_df['EMA26']

		# Calculate the Signal Line (9-day EMA of MACD)
		finance_df['Signal_Line'] = finance_df['MACD'].ewm(span=9, adjust=False).mean()



		###RSI
		# Calculate the daily price changes
		finance_df['Price_Change'] = finance_df['Close'].diff()

		# Separate gains and losses
		finance_df['Gain'] = finance_df['Price_Change'].apply(lambda x: x if x > 0 else 0)
		finance_df['Loss'] = finance_df['Price_Change'].apply(lambda x: -x if x < 0 else 0)

		# Calculate the average gain and loss over a 14-day period
		period = 14
		finance_df['Avg_Gain'] = finance_df['Gain'].rolling(window=period, min_periods=1).mean()
		finance_df['Avg_Loss'] = finance_df['Loss'].rolling(window=period, min_periods=1).mean()

		# Calculate the relative strength (RS)
		finance_df['RS'] = finance_df['Avg_Gain'] / finance_df['Avg_Loss']

		# Calculate the RSI
		finance_df['RSI'] = 100 - (100 / (1 + finance_df['RS']))


		####Used to calculate MFI
		# Step 1: Calculate the Typical Price (TP)
		finance_df['TP'] = (finance_df['High'] + finance_df['Low'] + finance_df['Close']) / 3

		# Step 2: Calculate the Money Flow (MF)
		finance_df['MF'] = finance_df['TP'] * finance_df['Volume']

		# Step 3: Calculate Positive and Negative Money Flow
		finance_df['Positive_MF'] = finance_df['MF'].where(finance_df['TP'] > finance_df['TP'].shift(1), 0)
		finance_df['Negative_MF'] = finance_df['MF'].where(finance_df['TP'] < finance_df['TP'].shift(1), 0)

		# Step 4: Calculate the rolling sum of Positive and Negative Money Flow over the specified period (e.g., 14 periods)
		window = 14
		finance_df['Positive_MF_sum'] = finance_df['Positive_MF'].rolling(window=window).sum()
		finance_df['Negative_MF_sum'] = finance_df['Negative_MF'].rolling(window=window).sum()

		# Step 5: Calculate the Money Flow Ratio
		finance_df['Money_Flow_Ratio'] = finance_df['Positive_MF_sum'] / finance_df['Negative_MF_sum']

		# Step 6: Calculate the Money Flow Index (MFI)
		finance_df['MFI'] = 100 - (100 / (1 + finance_df['Money_Flow_Ratio']))


		###Normalize stock price variables
		#normalize via log transform
		finance_df['Log_Close'] = np.log(finance_df['Close'])
		finance_df['Log_Open'] = np.log(finance_df['Open'])
		finance_df['Log_High'] = np.log(finance_df['High'])
		finance_df['Log_Low'] = np.log(finance_df['Low'])

		#normalize via Sklearn's scaler function
		scaler_close = MinMaxScaler()
		scaler_open = MinMaxScaler()
		scaler_high = MinMaxScaler()
		scaler_low = MinMaxScaler()
		finance_df['Normalized_Close'] = scaler_close.fit_transform(finance_df[['Close']])
		finance_df['Normalized_Open'] = scaler_open.fit_transform(finance_df[['Open']])
		finance_df['Normalized_High'] = scaler_high.fit_transform(finance_df[['High']])
		finance_df['Normalized_Low'] = scaler_low.fit_transform(finance_df[['Low']])

		# Store the parameters of the scaler in the session; this is for views.py
		session['scaler_open_min'] = scaler_open.min_.tolist()
		session['scaler_open_scale'] = scaler_open.scale_.tolist()
		session['scaler_close_min'] = scaler_close.min_.tolist()
		session['scaler_close_scale'] = scaler_close.scale_.tolist()
		session['scaler_high_min'] = scaler_high.min_.tolist()
		session['scaler_high_scale'] = scaler_high.scale_.tolist()
		session['scaler_low_min'] = scaler_low.min_.tolist()
		session['scaler_low_scale'] = scaler_low.scale_.tolist()



		#exclude the first 26 rows because calculations from MACD needs at least 26 days to calculate
		finance_df = finance_df.iloc[26:]
		first_row_date = str(finance_df.index[0])
		first_row_date = re.sub(r'\s.*$', '', first_row_date)
		finance_df = finance_df.reset_index(drop=True)
	



		####Create another new column 'Random_Yes_No_2' to test random values of 'yes' to simulate presence of a random pattern; only applicable to the "Random" pattern

		#Subset data frame for desired pattern
		if (pattern == "Random"):

			#num_yes = 2200
			num_yes = int(float(len(finance_df) * 0.35)) #yes values would just be used for training and validation since it uses stratified cross validation
			if (num_yes >= 2200): #set the max number of training observations, so training isn't too slow
				num_yes = 2200
			num_test = int(float(len(finance_df) * 0.35)) #yes_test values, holdout these values later as they would be used for testing
			if (num_test >= 2200): #set the max number of training observations
				num_test = 2200

			# Create a list of "Yes", "No", and "Yes_test" values. "Yes" used for training/validation, "No" are discarded observations/not used in the analysis. "Yes_test" used for observations that will be held out of training/validation to be later used for testing
			yes_no_list = (["Yes"] * num_yes) + (["No"] * (len(finance_df) - (num_yes + num_test))) + (["Yes_test"] * num_test)

			#set seed for reproducibility
			np.random.seed(6) 

			# Shuffle the list to randomize the order
			np.random.shuffle(yes_no_list)

			# Add the list as a new column in the DataFrame
			finance_df['Random_Yes_No_2'] = yes_no_list

			#Set the value of pattern_df to equal "Yes" values for the "Random" candlestick pattern
			#Pattern_df has a column "Row_index" which will contain the indexes to create the data needed for indepedent and dependent datasets		
			pattern_df = finance_df[finance_df['Random_Yes_No_2'] == "Yes"]
		elif (pattern == "Hammer"): #If the pattern is "Hammer", set pattern_df to equal only where the hammer pattern is present
			pattern_df = finance_df[finance_df['Hammer_pattern'] == "Yes"]
		else: #If the pattern is "Inverted Hammer", set pattern_df to equal only where the inverted hammer pattern is present
			pattern_df = finance_df[finance_df['InvertedHammer_pattern'] == "Yes"]

		#How many days after the pattern is identified to use for the dependent variable
		days_out = float(days_out)

		#What percent increase from the current price is considered a positive class. For example 1.01 = 1% increase; 100 * 1.01 = 101. So if original price is $100, anything greater than $101 is considered a positive class.
		pct_increase = float(pct_increase)

		#Gather independent variables
		independent_list15 = []

		#gather dependent variables
		dependent_list = []

		#get the row indexes to where the desired candlestick pattern is present, where it be either the "Random", "Hammer", or "Inverted Hammer" patterns
		pattern_index = list(pattern_df["Row_index"])

		#Loop through those indexes to gather independent and dependent variable datasets
		for i in pattern_index:
			
			#unable to get 30 days worth of data if index is less than 56, because previously removed first 26 observations
			if (i < 56):
				continue

			#get 30 days worth of data to gather data for indpendent variables
			subset_df = finance_df[(finance_df["Row_index"] >= (i - 29)) & (finance_df["Row_index"] <= (i))]
			
			#Get day after data to gather closing price for dependent variable
			dependent_df = finance_df[finance_df["Row_index"] == (i)]
			dependent2_df = finance_df[finance_df["Row_index"] == (i + days_out)]
			
			temp_list15 = [] #initialize

			#append temp_list to independent_list
			if len(dependent2_df) > 0: #dependent2_df may have length of zero as it is a future date, data may not be available
			

				for index, row in subset_df.iterrows():
						
						test_array15 = np.array([row['Normalized_Open'], row['Normalized_Close'], row['Normalized_High'], row['Normalized_Low'], row['RSI'], row['MFI'], row['MACD'], row['Signal_Line']])
					
						temp_list15.append(test_array15)
						
				independent_list15.append(temp_list15)
			
				if (dependent2_df['Close'].iloc[0] > dependent_df['Close'].iloc[0] * pct_increase):
					dependent_list.append(1)
				else:
					dependent_list.append(0)

		independent_array15 = np.array(independent_list15)
		dependent_array = np.array(dependent_list)

		# Define the LSTM classification model
		def create_lstm_classification(input_shape):
			model = Sequential()
			
			# LSTM layers
			model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape))
			model.add(Dropout(0.2))  # Dropout to reduce overfitting
			
			model.add(LSTM(64, activation='tanh', return_sequences=False))  # Final LSTM layer
			model.add(Dropout(0.2))
			
			# Dense output layer for binary classification
			model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (probability)
			
			# Compile the model
			model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Binary cross-entropy for classification
			
			return model

		# Define independent and dependent variables
		X = independent_array15  # Shape: (890, 30, 8) -- Independent array 15 has the shape of (30, 8)
		y = dependent_array  # Shape: (890,) (binary labels, 0 or 1)
		total_observations_train_and_val = len(X) #Will use this to pass onto html form

		# Count the number of true (1) and false (0) labels
		label_counts = pd.Series(y).value_counts()

		# Get counts of True (1) and False (0) specifically
		total_true_labels = label_counts.get(1, 0)  # Will return 0 if no '1' exists; pass this to html form via jinja
		total_false_labels = label_counts.get(0, 0)  # Will return 0 if no '0' exists; pass this to html form via jinja

		
		
		###Now that the indpendent and dependent datasets are gathered, I can train my model, however...
		#if there is a previously trained model with same parameters, you can skip this part
		if (previously_trained_message == None):

			# Initialize list to hold models and their weights
			models = []

			# Loop for training 9 identical models
			for model_num in range(9):
				print(f"Training model {model_num + 1}/9...")

				# Initialize the KFold
				kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=(model_num+1))

				# Create and compile the model once per model_num (outside of the KFold loop)
				# This model will be reused for each fold
				input_shape = (30, X.shape[2])  # Define input shape based on your data
				model = create_lstm_classification(input_shape)

				# Initialize an empty list for storing fold accuracies
				fold_accuracies = []

				# Set up a ModelCheckpoint callback to save the model's weights when validation accuracy is improved
				checkpoint_path = f'trained_models/modelID_{neural_network_id}_modelNumber_{model_num + 1}_best_weights.keras' 
				checkpoint = ModelCheckpoint(checkpoint_path, 
											save_best_only=True, 
											monitor='val_accuracy', 
											mode='max', 
											verbose=1)

				K_fold_counter = 0
				# Perform stratified K-fold cross-validation
				for train_index, val_index in kf.split(X, y):
					print(f'Model: {model_num + 1}; K-fold: {(K_fold_counter + 1)}')
					
					            
					K_fold_counter = K_fold_counter + 1
					
					X_train, X_val = X[train_index], X[val_index]
					y_train, y_val = y[train_index], y[val_index]

					# Train the model  !!!!!!!Epochs should be set to 10; changed to 1 temporarily during QA with app
					model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[checkpoint])
					
				#Clear memory from model after training each model to free up space
				K.clear_session()
				gc.collect()


				# After training, load the best weights saved by ModelCheckpoint
				#best_weights_model = load_model(checkpoint_path)

				# Extract the weights as a binary stream (this will allow saving the weights in the database)
				#weights_stream = best_weights_model.get_weights()  # This returns a list of numpy arrays (weights)

				# Convert the weights to a format that can be stored in the database
				#weights_pickle = pickle.dumps(weights_stream)  # Serialize weights to binary format

				# Check if the DataFrame is empty (used for getting the ID associated to the model weights that will be trained and saved later)
				#if len(model_table_df_copy) == 0:  
				#	search_network_id = 1  # If no rows, set the 'search_network_id' to 1
				#else:
				#	search_network_id = model_table_df_copy['id'].iloc[-1] + 1  # Get the next available ID by adding 1 to the last 'id' in the model_table_df_copy. This is because the 'neural_network' table shows previously trained models, but the ID doesn't exist yet at this point in that table (since the form submission hasn't been completed). We need to append this ID to the 'trained_model' table for each of the 9 models' weights being saved.

				# Write trained_model object to trained_model table
				#new_model = trained_model(
				#	neural_network_id=int(search_network_id), 
				#	model_name=f'model_{model_num + 1}_best_weights.keras', 
				#	weights=weights_pickle  # Store the serialized weights in the 'weights' field
				#)

				# Add the new network to the session and commit the transaction to the database
				#db.session.add(new_model)
				#db.session.commit()

				# Optionally, remove the temporary model file after storing the weights (to save disk space)
				#os.remove(checkpoint_path)





		###Now at this point, we are running our trained models on the test dataset!!!
		#message = "Almost done! Just a few more seconds... Please don't refresh the page :)"
		#socketio.emit('training_update', {'message': message})  # Emit message to client

		#Subset data frame for desired pattern
		if (pattern == "Random"):
			pattern_df = finance_df[finance_df['Random_Yes_No_2'] == "Yes_test"]
		elif (pattern == "Hammer"):
			pattern_df = finance_df[finance_df['Hammer_pattern'] == "Yes_test"]
		else:
			pattern_df = finance_df[finance_df['InvertedHammer_pattern'] == "Yes_test"]

		#How many days after the pattern is identified to use for the dependent variable
		days_out = days_out

		#What percent increase from the current price is considered a positive class. For example 1.01 = 1% increase; 100 * 1.01 = 101. So if original price is $100, anything greater than $101 is considered a positive class.
		pct_increase = pct_increase

		#Gather independent variables
		independent_list15 = []

		#gather dependent variables
		dependent_list = []

		pattern_index = list(pattern_df["Row_index"])
		
		for i in pattern_index:

			#unable to get 30 days worth of data if index is less than 56, because previously removed first 26 observations
			if (i < 56):
				continue

			#get 30 days worth of data to gather data for indpendent variables
			subset_df = finance_df[(finance_df["Row_index"] >= (i - 29)) & (finance_df["Row_index"] <= (i))]
			
			#Get day after data to gather closing price for dependent variable
			dependent_df = finance_df[finance_df["Row_index"] == (i)]
			dependent2_df = finance_df[finance_df["Row_index"] == (i + days_out)]
			
			temp_list15 = []

			#append temp_list to independent_list
			if len(dependent2_df) > 0: #dependent2_df may have length of zero as it is a future date, data may not be available
			

				for index, row in subset_df.iterrows():
						
						test_array15 = np.array([row['Normalized_Open'], row['Normalized_Close'], row['Normalized_High'], row['Normalized_Low'], row['RSI'], row['MFI'], row['MACD'], row['Signal_Line']])
					
						temp_list15.append(test_array15)
						
				independent_list15.append(temp_list15)
			
				if (dependent2_df['Close'].iloc[0] > dependent_df['Close'].iloc[0] * pct_increase):
					dependent_list.append(1)
				else:
					dependent_list.append(0)

		independent_array15 = np.array(independent_list15)
		dependent_array = np.array(dependent_list)
		X = independent_array15
		y = dependent_array

	
		# Initialize empty list to load models into
		models = []
		
		#if there is a previously trained model with same parameters, then load that up, else use the ones that were just trained
		#if (previously_trained_message != None): #load up the previously trained model
		#	get_model_id = previously_trained_df['id'].iloc[0]
			#model_query_df = pd.read_sql_query(f"SELECT * FROM trained_model WHERE neural_network_id = {get_model_id}", db.get_engine())

			# Load models and weights (adjust the range to include all your models); loops through all 9 rows of model_query_df; each row has trained model weights that need to be retrieved
		#	for index, row in model_query_df.iterrows():
		#		input_shape = (30, X.shape[2])
		#		best_model = create_lstm_classification(input_shape)
		#		best_model.load_weights(f'./trained_models/modelID_{previously_trained_df["id"].iloc[0]}_modelNumber_{model_num + 1}_best_weights.keras')
				# Deserialize the weights from the 'weights' column (stored as binary data)
				#weights = pickle.loads(row['weights'])
				# Set the weights of the model
				#best_model.set_weights(weights)
		#		models.append(best_model)

		#else: #else, use the model that was just trained upon form submission; meaning there was no previously trained model

		#	# Load models and weights (adjust the range to include all your models)
		#	for model_num in range(0, 9):  # Change this range to include more models if necessary
		#		best_model = create_lstm_classification(input_shape)
		#		best_model.load_weights(f'model_{model_num + 1}_best_weights.keras')
		#		models.append(best_model)

		for model_num in range(0, 9):  # Change this range to include more models if necessary
			input_shape = (30, X.shape[2])
			best_model = create_lstm_classification(input_shape)
			best_model.load_weights(f'trained_models/modelID_{neural_network_id}_modelNumber_{model_num + 1}_best_weights.keras')
			models.append(best_model)


		# The testing sets
		X_test_final = X
		y_test_final = y

		# Initialize a list to store binary predictions from each model
		ensemble_predictions = []

		#initialize to get votes from each model
		votes = np.zeros(len(X_test_final), dtype=int)

		# Generate predictions from each model
		for model in models:
			# Get the predictions from the model (probabilities)
			predictions = model.predict(X_test_final)
			
			# Convert predictions to binary (0 or 1)
			binary_predictions = (predictions > 0.5).astype(int)  # Convert to 0 or 1

			counter = 0
			for value in binary_predictions: #for each value in the model's predictions, if the value is predicted 0, then no votes added, if 1, then add 1 vote
				if (value == 0):
					votes[counter] = votes[counter] + 0
				else:
					votes[counter] = votes[counter] + 1
				counter = counter + 1

		#because there are 9 different models, if the vote is greater than 5, then it is a positive class
		results = []
		for vote in votes:
			if (vote >= 5):
				results.append(1)
			else:
				results.append(0)

		# Create a pandas Series for comparison
		comparison = pd.Series(results == y_test_final)

		# Use value_counts() to count True/False occurrences
		counts = comparison.value_counts()
		num_true = counts.get(True, 0)  # 0 if True is not found
		num_false = counts.get(False, 0)  # 0 if False is not found
		#counts_2 = pd.Series(dependent_array).value_counts()

		# Convert results and y_test_final to numpy arrays for easier handling if they aren't already
		results = np.array(results)
		y_test_final = np.array(y_test_final)

		# Count the number of true (1) and false (0) labels
		label_counts = pd.Series(y_test_final).value_counts()

		# Get counts of True (1) and False (0) specifically
		total_true_labels_test = label_counts.get(1, 0)  # Will return 0 if no '1' exists; pass this to html form
		total_false_labels_test = label_counts.get(0, 0)  # Will return 0 if no '0' exists; pass this to html form

		# Compute confusion matrix
		tn, fp, fn, tp = confusion_matrix(y_test_final, results).ravel()

		#compute accuracy and most_frequent_class_test_pct, so can write out to DB
		accuracy = float(num_true) / (len(y_test_final))
		if (total_true_labels_test > total_false_labels_test):
			most_frequent_class_test_pct = float(total_true_labels_test) / float(len(y_test_final))
		else:
			most_frequent_class_test_pct = float(total_false_labels_test) / float(len(y_test_final))
		pattern=str(pattern)
		
		
		#skip this if there was already a previously trained model
		if (previously_trained_message == None):
			#Write neural_network object to neural_network table
			new_network = neural_network(ticker=ticker_query, pattern=pattern, days_out_parameter=days_out, percent_increase_parameter=pct_increase, total_observations_train_and_val=total_observations_train_and_val, accuracy_test=accuracy, most_frequent_class_test_pct=most_frequent_class_test_pct, data_start_date=first_row_date, data_end_date=data_end_date)

			# Add the new network to the session and commit the transaction to the database
			db.session.add(new_network)
			db.session.commit()

		# Return a JSON response that includes the redirect URL
		#return jsonify({
        #    'redirect_url': url_for('view_networks.RETRIEVE_RESULTS', ticker_query=ticker_query, pattern=pattern, days_out=days_out, pct_increase=pct_increase, data_end_date=data_end_date, total_observations=total_observations_train_and_val, total_false_labels=total_false_labels, total_true_labels=total_true_labels, total_observations_test=len(y_test_final), total_incorrect_test=num_false, total_correct_test=num_true, total_false_labels_test=total_false_labels_test, total_true_labels_test=total_true_labels_test, True_Negatives=tn, False_Positives=fp, False_Negatives=fn, True_Positives=tp)
        #})

		# Redirect to another view which will return the search results if search button is clicked
		return redirect(url_for('view_networks.RETRIEVE_RESULTS', ticker_query=ticker_query, pattern=pattern, days_out=days_out, pct_increase=pct_increase, data_end_date=data_end_date, total_observations=total_observations_train_and_val, total_false_labels=total_false_labels, total_true_labels=total_true_labels, total_observations_test=len(y_test_final), total_incorrect_test=num_false, total_correct_test=num_true, total_false_labels_test=total_false_labels_test, total_true_labels_test=total_true_labels_test, True_Negatives=tn, False_Positives=fp, False_Negatives=fn, True_Positives=tp))

	#loads up the home page
	return render_template('home.html', form=form, model_table_df=model_table_df)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)