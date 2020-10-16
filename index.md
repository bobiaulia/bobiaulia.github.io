# Welcome to Bobi's GitHub Pages

In this page I will share my machine learning projects.
If you interested in any project, feel free to discuss it with me.

## Thailand's Early covid-19 confirmed case forecasting using Timeseries (overview)

We all know about covid-19 spreading speed is very fast.
In its early encounter, many country try to surpress the its spreading speed.
However, it's still hard to predict how fast it will spread across the country.
This project objective is to get predictions or forecasting total confirmed covid-19 case in a country (I choosed Thailand).

### Data source

I get my data from [HERE](https://www.kaggle.com/imdevskp/corona-virus-report) (https://www.kaggle.com/imdevskp/corona-virus-report), and the one I used is covid_19_clean_complete.csv

### What will I do with the data?

From my prespective, we cas see this data as timeseries problem.
Despite timeseries is not the most suitable model to solve this problem, it still gave us some insight about what will happen in the future.

### Research method

First, I used python for my entire research.
Second, becasue I see this as a timeseries problem, I will compare the effectiveness of some deep learning architecture to solve this problem. In the process, I compred 5 type of LSTM architecture for timeseries
Lastly, after analyzing and choosing for the best architecture, I wrote small interactive code in python to show the prediction of # covid-19 confirmed case in Thailand based on my model

### Result

- **Best architecture in my experiment** <br/>

```python
# CNN LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, LSTM, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

block = 50

lstm_cnn = Sequential()
lstm_cnn.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
lstm_cnn.add(TimeDistributed(MaxPooling1D(pool_size=2)))
lstm_cnn.add(TimeDistributed(Flatten()))
lstm_cnn.add(LSTM(block, activation='relu'))
lstm_cnn.add(Dense(1))
lstm_cnn.compile(optimizer='adam', loss='mean_squared_error', metrics='cosine_similarity')
```
<br/>

- **Predicting (forecasting) method** <br/>

For some reason, I used rolling prediction (forecasting) method to predict the value with my model. The main reason is because LSTM is not sutable to predict far ahead in the future because of its instability.<br/>

- **Model application** <br/>

Here's my python code to apply the result! Just click > [Model Application](https://github.com/bobiaulia/Early-Covid-19-Project/blob/main/final_result/model_application.ipynb).<br/>
<br/>
How to run it in python? I hope this little guide will help.
  - Donwload the code and dataset (csv files except thailand_covid_19_cases.csv) in that folder.
  - Make sure the path for every files in python code in match with your files path in your directories.
  - Input your desired start and end date in the input sections.
  - Run all code.
  - You will get the graph of Tailand corfimed case in the range of your desired dates. The variable _desired_data_ contain series of dates with it's respective values of total confirmed cases
