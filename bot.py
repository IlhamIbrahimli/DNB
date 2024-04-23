import telebot
from config import *
from logic import *
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

bot = telebot.TeleBot(TOKEN)
manager = StockManager("user_settings.db")

manager.create_table("settings", "(user_id INTEGER, colour TEXT NOT NULL)")

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,"Welcome to the stock monitoring bot. Use /help to list commands")
    manager.first_config(message.chat.id)


@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id,"/set_color - Can set the colour of the graph line for each user seperately.(Usage: /set_color arg) If arg is left empty a list of available colours will appear.")
    bot.send_message(message.chat.id,"/help - Show this menu.")
    bot.send_message(message.chat.id,"/predict_NASDAQ - Trains an AI in realtime to predict the stock of a NASDAQ company. (Example Usage: /predict_NASDAQ MSFT)")
    bot.send_message(message.chat.id,"/history_NASDAQ - Get history of the stocks of a NASDAQ company over the last 250 days. (Example Usage: /history_NASDAQ MSFT)")


@bot.message_handler(commands=['predict_NASDAQ'])
def predict(message):

    input_text = telebot.util.extract_arguments(message.text)

    try:
        nasdaq = yf.Ticker(input_text)
    except:
        bot.send_message(message.chat.id,"No such NASDAQ name.")
    info = nasdaq.info
    if len(info) <= 1:
        bot.send_message(message.chat.id,"No such NASDAQ name.")
        
    else:
        try:
            bot.send_message(message.chat.id,"AI is training...")
            data,length = manager.seq_model(input_text)
            print(data) 
            print(type(data))
            file = manager.draw_graph(length,message.chat.id,data,f"Prediction for {input_text} in the next 60 days",True)

            bot.send_photo(message.chat.id, open(file,"rb"))
        except:
            bot.send_photo(message.chat.id, open(file,"rb"))



@bot.message_handler(commands=['history_NASDAQ'])
def history(message):

    input_text = telebot.util.extract_arguments(message.text)

    try:
        nasdaq = yf.Ticker(input_text)
    except:
        bot.send_message(message.chat.id,"No such NASDAQ name.")

    info = nasdaq.info
    if len(info) < 1:
        bot.send_message(message.chat.id,"No such NASDAQ name.")
        
    else:
        bot.send_message(message.chat.id,f"Getting stock data for {input_text}...")
        data,length = manager.get_data(input_text)
        proccessed = manager.proccess_data(data)

        file = manager.draw_graph(length,message.chat.id,proccessed,f"History of {input_text} in the last {length} days",False)

        bot.send_photo(message.chat.id, open(file,"rb"))


@bot.message_handler(commands=['set_color'])
def set_user_colour(message):
    input_text = telebot.util.extract_arguments(message.text)

    valid_colours = ["b","g","r","c","m","y","k","w"]

    if input_text not in valid_colours:
        bot.send_message(message.chat.id,"Not a valid colour.")
        bot.send_message(message.chat.id,"List of available colours:")
        bot.send_message(message.chat.id,"b: blue")
        bot.send_message(message.chat.id,"g: green")
        bot.send_message(message.chat.id,"r: red")
        bot.send_message(message.chat.id,"c: cyan")
        bot.send_message(message.chat.id,"m: magenta")
        bot.send_message(message.chat.id,"y: yellow")
        bot.send_message(message.chat.id,"k: black")
        bot.send_message(message.chat.id,"w: white")
    
    else:
        manager.set_colour(input_text,message.chat.id)
        bot.send_message(message.chat.id, "Colour has been saved!")
    
    
    

bot.infinity_polling(timeout=10000, none_stop=True)
