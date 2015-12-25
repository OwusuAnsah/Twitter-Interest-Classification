import pandas,csv,re,random,tweepy,datetime
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from accesstokenTwitter import * # python file containing a list of access tokens
from collections import Counter

test_data = pandas.read_csv('test_data.csv') # csv file with 2 headers: name, interest
Count_Vectorizer, Tfidf_Transformer, starttime, cachedStopwords, myStopwords, snowball, porter, selector, lancaster = None, None, None, None, None, None, None, None, None

# Call this function to run the program. It takes in 2 variables.
# First, a pandas dataframe defined above, which reads a csv file
# Second, a boolean which determines if the classifiers are loaded from file, or trained from scratch
def run(loadFromSave):
	dataframe = extractAllTweets(d=None) # createUserDict() returns a dictionary
	print("Initializing variables and environment")
	initializeCV()
	tfidf_doc, interestLabels = getFeatureVectorAndLabels(dataframe,selectFeatures = True) # true: feature selection ON and vice versa

	if loadFromSave:
		# joblib is an sklearn library that allows us to save / load the trained classifiers
		NBClassifier = joblib.load('classifiers/naivebayes.pkl')
		SVMClassifier = joblib.load('classifiers/svm.pkl')
		LogRegr = joblib.load('classifiers/logregr.pkl')
		print("Classifiers loaded from file")
	else:
		NBClassifier = trainNB(tfidf_doc,interestLabels)
		joblib.dump(NBClassifier,'classifiers/naivebayes.pkl')
		LogRegr = trainLogRegr(tfidf_doc,interestLabels)
		joblib.dump(LogRegr,'classifiers/logregr.pkl')
		SVMClassifier = trainSVM(tfidf_doc,interestLabels,"sgd")
		joblib.dump(SVMClassifier,'classifiers/svm.pkl')
		# uncomment to train classifiers using other kernels (NOT RECOMMENDED)
		# SVMClassifier_linear = trainSVM(tfidf_doc,interestLabels,"linear")
		# SVMClassifier_poly = trainSVM(tfidf_doc,interestLabels,"poly")
		# SVMClassifier_rbf = trainSVM(tfidf_doc,interestLabels,"rbf")
		# SVMClassifier_sigmoid = trainSVM(tfidf_doc,interestLabels,"sigmoid")
		# joblib.dump(SVMClassifier_linear,'classifiers/svm_linear.pkl')
		# joblib.dump(SVMClassifier_poly,'classifiers/svm_poly.pkl')
		# joblib.dump(SVMClassifier_rbf,'classifiers/svm_rbf.pkl')
		# joblib.dump(SVMClassifier_sigmoid,'classifiers/svm_sigmoid.pkl')
	classifiers = [NBClassifier,SVMClassifier,LogRegr]

	printKFoldScore(NBClassifier,tfidf_doc,interestLabels,"NBClassifier")
	# uncomment to see kfold score for the other classifiers
	# printKFoldScore(SVMClassifier,tfidf_doc,interestLabels,"SVMClassifier")
	# printKFoldScore(LogRegr,tfidf_doc,interestLabels,"LogRegr")
	
	printMetrics(NBClassifier,tfidf_doc,interestLabels,"NBClassifier")
	# uncomment to see classification report and confusion matrix for the other classifiers
	# printMetrics(SVMClassifier,tfidf_doc,interestLabels,"SVMClassifier")
	# printMetrics(LogRegr,tfidf_doc,interestLabels,"LogRegr")

	testClassifier(classifiers,test_data)

	while True:
		print("Press Ctrl+D to exit")
		targetHandle = input("Please enter a valid twitter handle: ")
		targetInterest_NB = predictInterest(targetHandle,classifiers,400,selectFeatures = True)
		print(targetHandle + " => " + targetInterest_NB + " (NBClassifier)")

# returns a dictionary where the key values are the interest categories and values are a list of twitter handles
def createUserDict():
	d={}
	d['news'] = ['cnnbrk', 'nytimes', 'ReutersLive', 'BBCBreaking', 'BreakingNews']
	d['inspiration'] = ['DalaiLama', 'BrendonBurchard', 'mamagena', 'marcandangel', 'LamaSuryaDas']
	d['sports'] = ['espn', 'SportsCenter', 'NBA', 'foxsoccer', 'NFL']
	d['music'] = ['thedailyswarm','brooklynvegan','atlantamusic','gorillavsbear','idolator']
	d['fashion'] = ['bof','fashionista_com','glitterguide','twopointohLA','whowhatwear']
	d['gaming'] = ['IGN','Kotaku','Polygon','shacknews','gamespot']
	d['politics'] = ['potus','ezraklein','politicalwire','nprpolitics','senatus']
	d['tech'] = ['TheNextWeb','recode','TechCrunch','TechRepublic','Gigaom']
	d['finance'] = ['chrisadamsmkts','pimco','StockTwits','stlouisfed','markflowchatter']
	d['food'] = ['nytfood','Foodimentary','TestKitchen','seriouseats','epicurious']
	return d

def extractAllTweets(d):
	# if no function specified, read csv from file
	if d == None:
		print("Loading tweets from file")
		return pandas.read_csv('tweets_all.csv')
	# tweet extraction using official twitter API
	else:
		with open('extracted_tweets.csv','w') as data:
			w = csv.writer(data)
			w.writerow(['name','interest','text'])
			api = authTwitter(random.randrange(0,5))
			starttime = datetime.datetime.now()

			for key, value in d.items():
				print('######### Topic: '+str(key) + ' #########')
				for i in range(len(value)):
					print('Currently processing user: ' + value[i] + ' time taken: ' + str(datetime.datetime.now() - starttime))                
					listOfTweets = []
					batch = api.user_timeline(screen_name = value[i],count=200)
					listOfTweets.extend(batch)
					lastId = listOfTweets[-1].id - 1
					counter = 16
					
					while len(batch) > 0 and counter > 1:
						counter -= 1
						batch = api.user_timeline(screen_name = value[i],count=200,max_id=lastId)
						listOfTweets.extend(batch)
						lastId = listOfTweets[-1].id - 1
					listOfTweets = [removeUrl(tweet.text) for tweet in listOfTweets]
						
					for tweet in listOfTweets:
						w.writerow([value[i],str(key),tweet])
		return pandas.read_csv('extracted_tweets')

# defines the required global variables, as well as stopwords or stemmers required
def initializeCV():
	global Count_Vectorizer, cachedStopwords, myStopwords, snowball, porter, selector, lancaster
	Count_Vectorizer = CountVectorizer(ngram_range=(1,3)) # change settings for unigram, bigram, trigram
	cachedStopwords = stopwords.words("english")
	snowball = SnowballStemmer("english")
	porter = PorterStemmer()
	lancaster = LancasterStemmer() # for testing purposes only
	selector = SelectKBest(f_classif, k=200000) # select top k features using f_classif/chi2 algo

# takes in a pandas dataframe and returns a feature vector (sparse matrix) and list of interests which act as labels
def getFeatureVectorAndLabels(dataframe,selectFeatures):
	global Tfidf_Transformer, selector
	print("Converting dataframe to list of tweets and list of interests")
	listOfTweets, interestLabels = dfToTweetsAndInterests(dataframe)
	print("Stemming list of tweets")
	listOfTweets = stemList(porter,listOfTweets) # change stemming algorithm here
	data_counts = Count_Vectorizer.fit_transform(listOfTweets)
	if selectFeatures:
		print("Selecting top k features")
		temp_selector = selector.fit(data_counts,interestLabels)
		data_counts = temp_selector.transform(data_counts)
		# data_counts = selector.fit(data_counts, interestLabels).transform(data_counts)
	print("There are %s features" % data_counts.shape[1])
	temp_tfidf_transformer = TfidfTransformer(use_idf=True).fit(data_counts)
	Tfidf_Transformer = temp_tfidf_transformer
	tfidf_doc = TfidfTransformer(use_idf=True).fit_transform(data_counts)
	return tfidf_doc, interestLabels

# takes in a stemmer object defined in initalizeCV() and a list of strings to be stemmed
def stemList(stemmer, listOfTweets):
	if stemmer != None:
		stemmedTokens =[]
		for sentence in listOfTweets:
			tokens = sentence.split(' ')
			tokens = [stemmer.stem(token) for token in tokens if not token.isdigit()]
			stemmedTokens.append(tokens)
		listOfTweets = []
		for token in stemmedTokens:
			listOfTweets.append(" ".join(str(i) for i in token))
	return listOfTweets

# preprocesses a pandas dataframe to return a cleaned dataset without punctuation and stopwords
def dfToTweetsAndInterests(dataframe):
	listOfTweets = []
	interestLabels = []
	for i in dataframe.index:
		tweet = dataframe.text[i]
		interest = dataframe.interest[i]
		if type(tweet) == str:
			# remove punctuation
			tweet = re.sub(r'[^\w\s]','',tweet)
			# remove stopwords and change to lowercase
			tweet = ' '.join([word.lower() for word in tweet.split() if word not in cachedStopwords])
			listOfTweets.append(tweet)
			interestLabels.append(interest)
	return listOfTweets, interestLabels

# train multinomial naive bayes classifier with given features and labels
def trainNB(features,labels):
	starttime = datetime.datetime.now()
	clf = MultinomialNB().fit(features, labels)
	print("Time taken to train NBClassifier: " + str(datetime.datetime.now() - starttime))
	return clf

# train support vector machine classifier with given features and labels
def trainSVM(features,labels,kernelType):
	starttime = datetime.datetime.now()
	if kernelType=='sgd':
		clf = SGDClassifier().fit(features,labels)
	# kernelType can be linear, poly, rbf or sigmoid
	else:
		clf = SVC(kernel=kernelType).fit(features,labels)
	print("Time taken to train SVMClassifier(" + kernelType + ") : " + str(datetime.datetime.now() - starttime))
	return clf

# train logistic regression (maximum entropy) classifier with given features and labels
def trainLogRegr(features,labels):
	starttime = datetime.datetime.now()
	clf = LogisticRegression().fit(features,labels)
	print("Time taken to train LogisticRegression: " + str(datetime.datetime.now() - starttime))
	return clf

# takes in a valid twitter handle, list of classification algorithms and number of tweets (>=200)
# returns an interest prediction based on the majority prediction of the trained classifiers
def predictInterest(targetHandle,classifiers,numTweets,selectFeatures):
	listOfTweets = mineTweets(targetHandle,numTweets) # input number of tweets to pull as desired (>= 200)
	data_counts = Count_Vectorizer.transform(listOfTweets)
	if selectFeatures:
		temp_selector = selector
		data_counts = temp_selector.transform(data_counts)
	tfidf_doc = TfidfTransformer(use_idf=True).fit(data_counts).transform(data_counts)
	predictions={}
	for classifier in classifiers:
		predictedList = classifier.predict(tfidf_doc)
		numWords = (word for word in predictedList if word[:1])
		targetInterest = Counter(numWords).most_common(1)[0][0] # returns the interest that is predicted the most number of times
		if targetInterest in predictions:
			predictions[targetInterest] += 1
		else:
			predictions[targetInterest] = 1
	return max(predictions)

# calls the twitter API to return a number of tweets from a specified twitter handle
def mineTweets(targetHandle,numOfTweets):
	api = authTwitter(3)
	listOfTweets = []
	counter = numOfTweets // 200 # max number of tweets per request is 200
	print('Mining %s tweets from %s' % ((counter)*200, targetHandle))
	batch = api.user_timeline(screen_name = targetHandle,count=200)
	listOfTweets.extend(batch)
	lastId = listOfTweets[-1].id - 1
	while len(batch) > 0 and counter > 1:
		counter -= 1
		batch = api.user_timeline(screen_name = targetHandle,count=200,max_id=lastId)
		listOfTweets.extend(batch)
		lastId = listOfTweets[-1].id - 1
	listOfTweets = [removeUrl(tweet.text) for tweet in listOfTweets]
	return listOfTweets

def removeUrl(text):
	return re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

def authTwitter(keyNum):
	keys = accesstokenlist[keyNum]
	auth = tweepy.auth.OAuthHandler(keys[0], keys[1])
	auth.set_access_token(keys[2], keys[3])
	return tweepy.API(auth)

# uses the cross_val_score method to calculate the accuracy of a model using kfold cross validation, with cv being the number of folds
def printKFoldScore(classifier, features, labels, name):	
	kfold_score = cross_val_score(classifier, features, labels, cv=10)
	print("Accuracy for " + name +  ": " + str(kfold_score.mean()))

# takes a prediction using the desired classifier and prints the classification report and confusion matrix
def printMetrics(classifier,features,labels,name):
	predictedList = classifier.predict(features)
	print("Classification report for " + name)
	print(classification_report(labels, predictedList))
	cm = confusion_matrix(labels, predictedList)
	print(cm)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	np.set_printoptions(precision=2)
	plt.figure()
	plotMatrix(cm)
	plt.show()
	# uncomment to save confusion matrix to file
	# plt.savefig('confusion_matrix/%s Confusion Matrix.png' % name, bbox_inches='tight')


def plotMatrix(cm, title='Confusion matrix', cmap=plt.cm.YlOrRd):
	target_names=['fashion','finance','food','gaming','inspiration','music','news','politics','sports','tech'] # alphabetical order
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names, rotation=45)
	plt.yticks(tick_marks, target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

# runs a classifier against a manually curated dataset and calculates the accuracy
def testClassifier(classifiers,dataframe):
	interestLabels, predictedInterests, listOfHandles = [], [], []
	for i in dataframe.index:
		print("Processing twitter user %s/" % str(i+1) + str(len(dataframe)))
		twitterHandle = dataframe.name[i]
		interestLabels.append(dataframe.interest[i])
		predictedInterests.append(predictInterest(twitterHandle, classifiers, 400, selectFeatures = True))
		listOfHandles.append(twitterHandle)		
	if len(interestLabels) == len(predictedInterests):
		correct = 0
		print("False predictions:")
		for j in range(len(interestLabels)):
			if interestLabels[j] == predictedInterests[j]:
				correct += 1
			else:
				print(listOfHandles[j] + " is " + interestLabels[j] + " but predicted as " + predictedInterests[j])
		print(str(correct) + "/" + str(len(interestLabels)) + " users predicted correctly")

run(loadFromSave=True) # true if loading classifiers from file, false if retraining classifiers