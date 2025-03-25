import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath("intentsjson.json")
with open(file_path,"r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0,max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x,y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0
def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    menu=["Home","Conversation History","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == 'Home':
        st.write("welcome to the Chatbot. Please ask your queries and press enter after that!")
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv','w',newline = '',encoding = 'utf=-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input','Chat Response','Timestamp'])
        counter = counter + 1
        user_input = st.text_input("You:",key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:",value = response , height = 120 , max_chars = None,key=f"Chatbot_response_{counter}")
            timestamp = datetime.datetime.now().strftime("%Y-%M-%D %H:%M:%S")
            with open('chat_log.csv','a',newline='',encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str,response,timestamp])
            if response.lower() in ['goodbye','bye','take care']:
                st.write("Thnak you for chatting with me,have a spiffing day!")
                st.stop()
    elif choice == 'Conversation History':
        st.header("Conversation History")
        with open('chat_log.csv','r',encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"user:{row[0]}")
                st.text(f"Chatbot : {row[1]}")
                st.text(f"Timestamp:{row[2]}")
                st.markdown('---')
    elif choice == 'Abour':
        st.write("The goal of this project is to showcase that we create a chatbot using python with NLP,The Chatbot used logistic regression for the model and we used streamlit for the frontend")
        st.subheader("project Overview")
        st.write('''
        The project is divided into following parts :
        1.NLP techniques and Logistic Regression algorith is used to treain the chatbot on lebelled intents.
        2.To build the interface,we used streamlit as an interactive app''')
        st.subheader("Chatbot interface:")
        st.write("Chatbot interface is simple,it takes user input and returns the response from the chatbot after pressing the enter key")
        st.subheader("Conclusion")
        st.write("We build a chatbot using NLP and LR and used streamlit for the interactive app development .This project can be further more improved with more data, sophisticated NLP techniques ,lastly we can use Deep Learning")
if __name__== '__main__':
    main()
