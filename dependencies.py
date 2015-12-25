import pip

packages = ['numpy','scipy','matplotlib','pandas','tweepy','nltk','scikit-learn']

def install(package):
	pip.main(['install',package])

for package in packages:
	install(package)