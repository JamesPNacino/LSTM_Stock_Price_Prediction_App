from flask import Blueprint, render_template, redirect, url_for, make_response, session, jsonify, request
from myproject import db
from myproject.models import neural_network
from myproject.view_networks.forms import RunTickerModel_Form, Predict_Form, SearchCandlestick_Form
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler  # Used to normalize data
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
import sqlite3
import re
import pickle




#setup blueprint for the articles view
#Blueprint('view_articles') -- does not have to be the name of the folder, but makes it simpler if it is same name
networks_blueprint = Blueprint('view_networks',
                              __name__,
                              template_folder='templates')






@networks_blueprint.route('/model_results', methods=['GET', 'POST'])
def RETRIEVE_RESULTS():
	
    # Retrieve data from query parameters from app.py's INDEX() function, when the form was submitted
    ticker_query = request.args.get('ticker_query')
    pattern = request.args.get('pattern')
    days_out = request.args.get('days_out')
    pct_increase = request.args.get('pct_increase')
    data_end_date = request.args.get('data_end_date')

    total_observations = request.args.get('total_observations')
    total_false_labels = request.args.get('total_false_labels')
    total_true_labels = request.args.get('total_true_labels')
    total_observations_test = request.args.get('total_observations_test')
    total_incorrect_test = request.args.get('total_incorrect_test')
    total_correct_test = request.args.get('total_correct_test')
    total_false_labels_test = request.args.get('total_false_labels_test')
    total_true_labels_test = request.args.get('total_true_labels_test')
    True_Negatives = request.args.get('True_Negatives')
    False_Positives = request.args.get('False_Positives')
    False_Negatives = request.args.get('False_Negatives')
    True_Positives = request.args.get('True_Positives')

    #Calculate fields
    if (float(total_true_labels) > float(total_false_labels)):
        Majority_class_pct = f"{float(total_true_labels) / float(total_observations) * 100}%"
    else:
        Majority_class_pct = f"{float(total_false_labels) / float(total_observations) * 100}%"

    if (float(total_true_labels_test) > float(total_false_labels_test)):
        Majority_test_class_pct = f"{float(total_true_labels_test) / float(total_observations_test) * 100}%"
    else:
        Majority_test_class_pct = f"{float(total_false_labels_test) / float(total_observations_test) * 100}%"

    Accuracy = f"{float(total_correct_test) / float(total_observations_test) * 100}%"

    #Load flask form - the predict form just includes a button
    form = Predict_Form()

	# if button is clicked on the form
    if form.validate_on_submit():
        
        ###Need to get the most current 30 days worth of formatted independent variable data as 30 days is one sequence
        # Get today's date
        today_date = datetime.today()
        formatted_today_date = today_date.strftime('%Y-%m-%d')


        ticker = ticker_query #The stock ticker for 'S&P 500'
        ticker_symbol = ticker #used for the training step in a later section as need to save ticker data
        ticker = yf.Ticker(ticker)

        # Define the custom date range (start and end dates in 'YYYY-MM-DD' format)
        start_date = "2000-01-01"

        # Fetch the historical data for the defined date range
        finance_df = ticker.history(start=start_date, end=formatted_today_date)
        #finance_df = ticker.history(period="3mo") #Get 10 years worth of data, #10y, #max


        finance_df = finance_df.drop('Dividends', axis=1) #remove 'Dividends' column
        finance_df = finance_df.drop('Stock Splits', axis=1) #remove 'Dividends' column
        if 'Capital Gains' in finance_df.columns:
            finance_df = finance_df.drop('Capital Gains', axis=1)
        finance_df['Row_index'] = range(1, len(finance_df) + 1) #creates index column


        #Creates a conditional statement based on conditions, and applies the 'choices' label.
        #This will result in a new column 'Candle', which describes if the daily candle was 'Bullish',
        #Bearish, or Neutral
        conditions = [
            finance_df['Close'] > finance_df['Open'],
            finance_df['Close'] == finance_df['Open'],
            finance_df['Open'] > finance_df['Close']
        ]
        choices = ['Bullish', 'Neutral', 'Bearish']
        finance_df['Candle'] = np.select(conditions, choices, default='Unknown')
        finance_df.head() 

        #Create column 'Body_length' which calculates the body length of the 
        #candle. 
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

        # Retrieve session data; Recreate the MinMaxScaler function from app.py
        scaler_low_min = np.array(session['scaler_low_min'])
        scaler_low_scale = np.array(session['scaler_low_scale'])
        scaler_low = MinMaxScaler()
        scaler_low.min_ = scaler_low_min
        scaler_low.scale_ = scaler_low_scale

        scaler_high_min = np.array(session['scaler_high_min'])
        scaler_high_scale = np.array(session['scaler_high_scale'])
        scaler_high = MinMaxScaler()
        scaler_high.min_ = scaler_high_min
        scaler_high.scale_ = scaler_high_scale

        scaler_close_min = np.array(session['scaler_close_min'])
        scaler_close_scale = np.array(session['scaler_close_scale'])
        scaler_close = MinMaxScaler()
        scaler_close.min_ = scaler_close_min
        scaler_close.scale_ = scaler_close_scale

        scaler_open_min = np.array(session['scaler_open_min'])
        scaler_open_scale = np.array(session['scaler_open_scale'])
        scaler_open = MinMaxScaler()
        scaler_open.min_ = scaler_open_min
        scaler_open.scale_ = scaler_open_scale



        #normalize via Sklearn's scaler function
        #scaler_close = MinMaxScaler()
        #scaler_open = MinMaxScaler()
        #scaler_high = MinMaxScaler()
        #scaler_low = MinMaxScaler()
        finance_df['Normalized_Close'] = scaler_close.transform(finance_df[['Close']].values)
        finance_df['Normalized_Open'] = scaler_open.transform(finance_df[['Open']].values)
        finance_df['Normalized_High'] = scaler_high.transform(finance_df[['High']].values)
        finance_df['Normalized_Low'] = scaler_low.transform(finance_df[['Low']].values)

        finance_df = finance_df.tail(30) #get the most recent 30 days worth of data

        #get the last date (the most recent date)
        last_row_name = str(finance_df.index[-1])
        last_row_name = re.sub(r'\s.*$', '', last_row_name)
        last_close = finance_df.Close[-1]


        #finance_df is only 30 rows here, so the last value in this column, mark it as "Yes"
        finance_df['Random_Yes_No_3'] = ['No'] * 29 + ['Yes'] 

        #Subset data frame for desired pattern
        pattern_df = finance_df[finance_df['Random_Yes_No_3'] == "Yes"]

        #How many days after the pattern is identified to use for the dependent variable
        days_out = int(float(days_out))

        #What percent increase from the current price is considered a positive class. For example 1.01 = 1% increase; 100 * 1.01 = 101. So if original price is $100, anything greater than $101 is considered a positive class.
        pct_increase = float(pct_increase)

        #Gather independent variables
        independent_list15 = []

        #gather dependent variables
        dependent_list = []

        #pattern_index only has one observation, get the indepedent variable information from this
        pattern_index = list(pattern_df["Row_index"])
        for i in pattern_index:

            #get 30 days worth of data to gather data for indpendent variables
            subset_df = finance_df[(finance_df["Row_index"] >= (i - 29)) & (finance_df["Row_index"] <= (i))]
            
            temp_list15 = []

            for index, row in subset_df.iterrows():
                    
                    test_array15 = np.array([row['Normalized_Open'], row['Normalized_Close'], row['Normalized_High'], row['Normalized_Low'], row['RSI'], row['MFI'], row['MACD'], row['Signal_Line']])
                
                    temp_list15.append(test_array15)
                    
            independent_list15.append(temp_list15)
            

        independent_array15 = np.array(independent_list15)
        X = independent_array15




        # Assuming you already have your trained models and their weights
        models = []

        # Define the LSTM classification model
        input_shape = (30, X.shape[2])
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


        #Get the data from the neural network model that was trained or loaded from app.py
        previously_trained_df = pd.read_sql_query(f"SELECT * FROM neural_network WHERE ticker='{ticker_query}' AND pattern='{pattern}' AND days_out_parameter='{days_out}' AND percent_increase_parameter='{pct_increase}'AND data_end_date='{data_end_date}'", db.get_engine())
        
        #Get the ID associated with that neural network model
        neural_network_id = int(previously_trained_df['id'].iloc[0])
       
        #Get the data from trained model where it equals the 'neural_network_id', this is where the model weights for the 9 models are located in this dataframe
        #model_query_df = pd.read_sql_query(f"SELECT * FROM trained_model WHERE neural_network_id = {neural_network_id}", db.get_engine())




        # Load models and weights (adjust the range to include all your models); loops through all 9 rows of model_query_df; each row has trained model weights that need to be retrieved
        for model_num in range(0, 9):  # Change this range to include more models if necessary
            best_model = create_lstm_classification(input_shape)
            best_model.load_weights(f'trained_models/modelID_{neural_network_id}_modelNumber_{model_num + 1}_best_weights.keras')
            models.append(best_model)


        # Initialize a list to store binary predictions from each model
        ensemble_predictions = []

        #initialize to get votes from each model
        votes = np.zeros(len(X), dtype=int)

        # Generate predictions from each model
        for model in models:
            # Get the predictions from the model (probabilities)
            predictions = model.predict(X)
            print(predictions)
            
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
        
        #some extra information for the user, if the prediction is a positive class, show them some string value, else...
        if (results[0] == 1):
            str_class = "positive"
            str_class2 = "be"
        else:
            str_class = "negative"
            str_class2 = "not be"


        #Pass these values to the template for rendering, but also to render the prediction information to show to user
        return render_template("model_results.html", form=form, ticker_query=ticker_query, pattern=pattern, days_out=days_out, pct_increase=pct_increase, data_end_date=data_end_date, total_observations=total_observations, total_false_labels=total_false_labels, total_true_labels=total_true_labels, total_observations_test=total_observations_test,total_incorrect_test=total_incorrect_test, total_correct_test=total_correct_test, total_false_labels_test=total_false_labels_test,total_true_labels_test=total_true_labels_test, True_Negatives=True_Negatives, False_Positives=False_Positives,False_Negatives=False_Negatives, True_Positives=True_Positives, Majority_class_pct=Majority_class_pct, Majority_test_class_pct=Majority_test_class_pct, Accuracy=Accuracy, results=results, votes=votes, last_row_date=last_row_name, last_close=round(last_close, 4), goal_price=round(last_close * pct_increase, 4), str_class=str_class, str_class2=str_class2)


    # Pass these values to the template for rendering
    return render_template("model_results.html", form=form, ticker_query=ticker_query, pattern=pattern, days_out=days_out, pct_increase=pct_increase, data_end_date=data_end_date, total_observations=total_observations, total_false_labels=total_false_labels, total_true_labels=total_true_labels, total_observations_test=total_observations_test,total_incorrect_test=total_incorrect_test, total_correct_test=total_correct_test, total_false_labels_test=total_false_labels_test,total_true_labels_test=total_true_labels_test, True_Negatives=True_Negatives, False_Positives=False_Positives,False_Negatives=False_Negatives, True_Positives=True_Positives, Majority_class_pct=Majority_class_pct, Majority_test_class_pct=Majority_test_class_pct, Accuracy=Accuracy)




@networks_blueprint.route('/search_candlesticks', methods=['GET', 'POST'])
def SEARCH_CANDLESTICKS():
    
    #initialize flask form
    form = SearchCandlestick_Form()

    #if form is validated upon submit
    if form.validate_on_submit():
    
        
        #get data from input box
        ticker_query = form.ticker_query.data.upper()

        # Get today's date
        today_date = datetime.today()
        formatted_today_date = today_date.strftime('%Y-%m-%d')


        ticker = ticker_query #The stock ticker
        ticker_symbol = ticker #used for the training step in a later section as need to save ticker data
        ticker = yf.Ticker(ticker)

        # Define the custom date range (start and end dates in 'YYYY-MM-DD' format)
        start_date = "2000-01-01"

        # Fetch the historical data for the defined date range
        finance_df = ticker.history(start=start_date, end=formatted_today_date)


        finance_df = finance_df.drop('Dividends', axis=1) #remove 'Dividends' column
        finance_df = finance_df.drop('Stock Splits', axis=1) #remove 'Dividends' column
        if 'Capital Gains' in finance_df.columns:
            finance_df = finance_df.drop('Capital Gains', axis=1)
        finance_df['Row_index'] = range(1, len(finance_df) + 1) #creates index column


        #Creates a conditional statement based on conditions, and applies the 'choices' label.
        #This will result in a new column 'Candle', which describes if the daily candle was 'Bullish',
        #Bearish, or Neutral
        conditions = [
            finance_df['Close'] > finance_df['Open'],
            finance_df['Close'] == finance_df['Open'],
            finance_df['Open'] > finance_df['Close']
        ]
        choices = ['Bullish', 'Neutral', 'Bearish']
        finance_df['Candle'] = np.select(conditions, choices, default='Unknown')
        finance_df.head() 

        #Create column 'Body_length' which calculates the body length of the 
        #candle. 
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

        finance_df = finance_df.tail(30) #get the most recent 30 days worth of data

        #get the last date (the most recent date)
        last_row_date = str(finance_df.index[-1])
        last_row_date = re.sub(r'\s.*$', '', last_row_date)
        hammer_pattern = finance_df.Hammer_pattern[-1]
        if (hammer_pattern == "No"):
            hammer_pattern = "not present"
        invertedhammer_pattern = finance_df.InvertedHammer_pattern[-1]
        if (invertedhammer_pattern == "No"):
            invertedhammer_pattern = "not present"

        return render_template("search_candlesticks.html", form=form, ticker_query=ticker_query, last_row_date=last_row_date, hammer_pattern=hammer_pattern, invertedhammer_pattern=invertedhammer_pattern)

    return render_template("search_candlesticks.html", form=form)





@networks_blueprint.route('/about_model', methods=['GET', 'POST'])
def ABOUT_MODEL():
    return render_template("about_model.html")