'Please keep in mind that Machine Learning (ML) is an entire course by itself.'
'This segment covers the overview of ML in the hope that you can comprehend'
'the subject matter and be able to explore more on your own'
'Heian Alrousan'
'Homework 10'

#Import Seaborn
#Import Matplotlib
import seaborn as Snsborn
import matplotlib.pyplot as plt

#15.3 (Seaborn Pairplot Graph)
iris = Snsborn.load_dataset("Iris") #Python package for data analysis and visualization in Earth Science
graphing = Snsborn.pairplot(iris) #Create a Seaborn pairplot graph (like we showed for Iris) 
plt.show() #Show Data

#15.6 (Seaborn Pairplot Graph)
#Reimplement the simple linear regression case study of Section 15.4 using the average yearly temperature data.
import pandas as pd
nyc = pd.read_csv('ave_yearly_temp_nyc_1895-2017.csv')
nyc.columns = ['Date', 'Value', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
graphing = Snsborn.pairplot(nyc)
plt.show()

#How does the temperature trend compare to the average January high temperatures?
