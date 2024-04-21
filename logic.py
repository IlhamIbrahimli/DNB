import numpy as np
import datetime as dt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import sqlite3


class StockManager:

    def __init__(self,db) :
        self.db = db

        
    def create_table(self,table_name,parameters: str):
        self.table_name = table_name
        con = sqlite3.connect(self.db)
        with con:
            cur = con.cursor()
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {parameters}")
            con.commit()
            cur.close()


    def first_config(self,user_id):
        con = sqlite3.connect(self.db)
        with con:
            cur = con.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE user_id = {user_id}")
            res = cur.fetchone()
            if res[0] <= 0:
                cur.execute(f"INSERT INTO {self.table_name} VALUES({user_id},'r')")



    def seq_model(self,stock_name):
        
        self.company = stock_name
        self.data = yf.download(self.company, start=dt.datetime(2018, 1, 1), end=dt.datetime.now())

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

        self.prediction_days = 120

        self.x_train = []
        self.y_train = []

        for x in range(self.prediction_days, len(self.scaled_data)):
            self.x_train.append(self.scaled_data[x - self.prediction_days:x, 0])
            self.y_train.append(self.scaled_data[x, 0])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        self.model = Sequential()

        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1],1)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=1)) #Prediction of the next closing value

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train,self.y_train,epochs=25,batch_size=32)

        self.test_start = dt.datetime.now()
        self.model_inputs = self.scaled_data[-self.prediction_days:].reshape(-1, 1)
        self.predicted_prices_future = []
        self.future_days = 60
        # Loop to predict future prices
        for _ in range(self.future_days):
            # Reshape the input data for the model
            self.x_future = self.model_inputs[-self.prediction_days:].reshape(1, self.prediction_days, 1)

            # Use the model to predict the next day's price
            self.predicted_price = self.model.predict(self.x_future)[0][0]

            self.predicted_price += np.random.normal(0, 0.02)

            # Append the predicted price to the list
            self.predicted_prices_future.append(self.predicted_price)

            # Update the input data for the next prediction
            self.model_inputs = np.append(self.model_inputs, self.predicted_price)

        # Inverse transform the predicted prices to the original scale
        self.predicted_prices_future = self.scaler.inverse_transform(np.array(self.predicted_prices_future).reshape(-1, 1))
        return (self.predicted_prices_future,self.future_days)

    def draw_graph(self,length,user_id,data,name,prediction):
        self.y_graph = []
        self.x_graph = []
        for i in range(length):
            
            if prediction:
                self.y_graph.append(data[i][0])
                self.x_graph.append(dt.date.today() + dt.timedelta(i))
            else:
                self.y_graph.append(data[i])
                self.x_graph.append(dt.date.today() - dt.timedelta(i))
        if prediction:
            
            file_name = f"{user_id}_prediction.png" 
        else: 
            file_name = f"{user_id}_history.png"
            self.x_graph.reverse()
        self.first_config(user_id)
        colour = self.get_colour(user_id)
        plt.figure(figsize=(len(self.x_graph)/10,max(self.y_graph)/10))
        plt.plot(self.x_graph,self.y_graph,marker='o',color=colour[0],linestyle= "-")
        plt.title(name)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=90)
        plt.grid(True)
        if prediction: file_name = f"{user_id}_prediction.png" 
        else: file_name = f"{user_id}_history.png"
        plt.savefig(file_name)
        return file_name
    
    def get_data(self,stock_name):
        self.company = stock_name

        self.data = yf.download(self.company, start=dt.date.today() - dt.timedelta(365), end=dt.date.today())
        print(self.data)
        return self.data,len(self.data)
    
    def proccess_data(self,data):
        self.column = data["Close"]
        self.proccess = self.column.tolist()
        print(self.proccess)
        print(len(self.proccess))

        return self.proccess
    
    def set_colour(self,colour,user_id):
        con = sqlite3.connect(self.db)
        with con:
            cur = con.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE user_id = {user_id}")
            res = cur.fetchone()
            if res[0] <= 0:
                cur.execute(f"INSERT INTO {self.table_name} VALUES({user_id},'r')")
            else:
                cur.execute(f"UPDATE {self.table_name} SET colour = '{colour}' WHERE user_id = {user_id}")

    def get_colour(self,user_id):
        con = sqlite3.connect(self.db)
        with con:
            cur = con.cursor()
            cur.execute(f"SELECT colour FROM {self.table_name} WHERE user_id = {user_id}")
            res = cur.fetchone()
            return res
    
    
