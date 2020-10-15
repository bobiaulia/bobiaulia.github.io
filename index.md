# Welcome to Bobi's GitHub Pages

In this page i will share my machine learning projects. If you interested in any project, feel free to discuss it with me.

## Thailand Early covid-19 confirmed case forecasting using Timeseries (overview)

We all know about covid-19 spreading speed is very fast. In its early encounter, many country try to surpress the its spreading speed. However, it's still hard to predict how fast it will spread across the country. This project objective is to get predictions or forecasting total confirmed covid-19 case in a country (i choosed Thailand).

### Data source

I get my data from [HERE](https://www.kaggle.com/imdevskp/corona-virus-report) (https://www.kaggle.com/imdevskp/corona-virus-report), and the one I used is covid_19_clean_complete.csv

### What will I do with the data?

From my prespective, we cas see this data as timeseries problem. Despite timeseries is not the most suitable model to solve this problem, it still gave us some insight about what will happen in the future.

### Research method

First, i used python for my entire research.
Second, becasue i see this as a timeseries problem, i will compare the effectiveness of some deep learning architecture to solve this problem. In the process, i compred 5 type of LSTM architecture for timeseries
Lastly, after analyzing and choosing for the best architecture, i wrote small interactive code in python to show the prediction of # covid-19 confirmed case in Thailand based on my model

### Result

- Best architecture in my experiment
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, LSTM, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
```
