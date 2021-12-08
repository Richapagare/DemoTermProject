from flask import Flask, render_template,request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import string

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/upload')
def upload():
	return render_template("upload.html")

@app.route('/query')
def query():
	return render_template("query.html")


@app.route('/predict',methods=['POST'])
def predict():
	print("Hello")
	df = pd.read_csv("imdb_labelled.txt",delimiter='\t', header=None, names=['text', 'label'])
	print(df)
	punct=string.punctuation
	corpus = []
	# i=0
	for sentence in df['text']:
		sentence = re.sub('[^a-zA-Z]',' ',str(sentence))
		sentence = re.sub('<[^<]+?>', ' ',str(sentence))
		sentence = sentence.lower()
		sentence = sentence.split()
		sentence = [PorterStemmer().stem(word) for word in sentence if word not in set(stopwords.words('english')) and word not in punct]
		corpus.append(' '.join(str(x) for x in sentence))
	y = df['label']
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(corpus)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
	clf= MultinomialNB()
	clf.fit(X_train, y_train)
	clf.score(X_test,y_test)
	
	if request.method=='POST':
		message = request.form['message']
		data = [message]
		print(data)
		vect = vectorizer.transform(data).toarray()
		my_prediction = clf.predict(vect)
		print(my_prediction)
	#str="Label associated :"
	return render_template('result.html',prediction = my_prediction)
	

if __name__== '__main__' :
	app.run(debug=True)

	


	

