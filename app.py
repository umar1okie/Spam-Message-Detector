import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenize

    y = []
    for i in text:
        if i.isalnum():  # keep only alphanumeric
            y.append(i)

    text = y[:]  # copy
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # ✅ join list back into a string


# Load TF-IDF vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Check"):
    if input_sms.strip() != "":
        # 1. Preprocessing
        transformed_sms = transform_text(input_sms)

        # 2. Vectorizing
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predicting
        result = model.predict(vector_input)[0]

        # 4. Displaying
        if result == 1:
            st.error("🚨 The message is **Spam**")
        else:
            st.success("✅ The message is **Not Spam**")
    else:
        st.warning("⚠️ Please enter a message before checking.")
