from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df["message"]
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test,y_test)
    # Alternative Usage of Saved Model

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)