#Natural language Processing toolkit in Python 
import nltk
from nltk.corpus import stopwords



#for the user input

while True:
    input=raw_input("Enter any sentence (or 'exit' to quit: ")
    if input=='exit':
        break
    else:
        input=input.lower()
        input=input.split()
        print '\nWe think that sentiment was '+classifier.classify(feature_extractor(input))+' in that sentence'

        
"""
'stopwords' is a list of words very commonly used in the English language
that add no real meaning to the sentence ('a', 'the', 'all', 'many' etc),
which will cloud the classifier if we allow them to be used for sentiment
analysis. Thus, we import a corpus of stopwords from the NLTK library that
allow us to quickly strip them. However, I would advise writing your own
list of stopwords, as my limited experience has found them to strip
words that do contain some meaning.
"""

#we are setting our own customwords
customwords=['band','they','them']

#Load positive tweets into a list
p=open('postweets.txt','r')
postxt=p.readlines()    #List of all the lines of postweets.txt file


#Load negative tweets into a list
n=open('negtweets.txt','r')
negtxt=n.readlines()     #List of all the lines of negtweets.txt file


# Rather than tagging each word with sentiment.We will be using "zip" function.which combines two lists into a list of tuples
neglist=[]
poslist=[]


#create a list of 'negatives' with the exact length of our negative tweet list
for i in range(0,len(negtxt)):
    neglist.append('negative')

#Again create a list of 'positives' with the exact length of our postive tweet list
for i in range(0,len(postxt)):
    poslist.append('positive')
    

#Creates a list of tuples, with sentiment tagged
postagged=zip(postxt,poslist)
negtagged=zip(negtxt,neglist)


#Combines all of the tagged tweets to one large list
taggedtweets=postagged+negtagged


#now we will work with the individual words in the tweets - namely, getting a list
#of all of the words in the tweets, and then ordering them by the frequency in which they appear. 
tweets=[]         #List

#Create a list of words in the tweet, within a tuple.
for (word,sentiment) in taggedtweets:
    word_filter=[i.lower() for i in word.split()]
    tweets.append((word_filter,sentiment))

#Pull out all of the words in a list of tagged tweets, formatted in tuples
def getwords(tweets):
    allwords=[]
    for (words,sentiment) in tweets:
        allwords.extend(words)  # extend() method appends the contents of seq to list
    return allwords

#Order a list of tweets by their frequency.
def getwordfeatures(listoftweets):
    #Print out wordfreq if you want to have a look at the individual counts of words.
    wordfreq=nltk.FreqDist(listoftweets)   #this fucntion always return in decreasing order
    words=wordfreq.keys()
    return words

#Calls above functions - gives us list of the words in the tweets, ordered by freq.
wordlist=getwordfeatures(getwords(tweets))

#removing stopwords and customwords from wordlist
wordlist=[i for i in wordlist if not i in stopwords.words('english')]
wordlist=[i for i in wordlist if not i in customwords]

def feature_extractor(doc):
    docwords=set(doc)
    features={}
    for i in wordlist:
        features['contains(%s)'%i]=(i in docwords)
    return features

#creates a training set-classifier learns distribution of true/falses in the input.
training_set=nltk.classify.apply_features(feature_extractor,tweets)

    
#Now, all that is left to do is for us to train our classifier on the training set we just created.
#Print out the training_set list before if you want to find out a little more. 
classifier=nltk.NaiveBayesClassifier.train(training_set)


