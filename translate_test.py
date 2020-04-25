# -*- coding: utf-8 -*-
"""NLP-SentimentAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zSaMqF72z3p9JwqOSQZr8d3NhYiUaTSg
"""


#nltk.downloader.download('vader_lexicon')



#pip install googletrans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googletrans import Translator
translator = Translator()
t_file=open('data/BaseLine/test_sentence.txt',encoding='utf-8')
text=t_file.read().split('\n')
for te in text:
    print(te)
    t = translator.translate(te,dest='en')
    print(t.text)

    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(t.text)
    print(sentiment)

#pip install sentencepiece
