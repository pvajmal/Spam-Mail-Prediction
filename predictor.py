# Load the model from the file
import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer



input_mail = ['''
Congratz! you won 3000 USD. Please call 323232d33333 to redeem your reward.

''']


predictMail = joblib.load('SPAMPred.pkl')
feature_extraction = joblib.load('FeaturePred.pkl')
input_mail_features = feature_extraction.transform(input_mail)
prediction_on_input_mail = predictMail.predict(input_mail_features))
if prediction_on_input_mail == 0:
   print("Spam Mail")
else:
  print('Ham Mail')
