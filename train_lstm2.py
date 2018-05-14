import json
import numpy as np
import dateutil.parser as dp

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", default=0, help="Show debug message 0:no message, 1:show predict, 2:show train+predict, 3: show plot")
ap.add_argument("-r", "--repeat", default=1, help="training times")
args = vars(ap.parse_args())

DEBUG = int(args['debug'])

code = '0050'
TRAIN_TEST = 0.8
# code = '2317'
# TRAIN_TEST = 0.8

name = 'tw{0}'.format(code)

FIELDS = []
# FIELDS.append('date')
# FIELDS.append('low')
# FIELDS.append('high')
# FIELDS.append('open')
FIELDS.append('close')
# FIELDS.append('change')
# FIELDS.append('transaction')
# FIELDS.append('turnover')
# FIELDS.append('capacity')

BLOCK_SIZE = 1
BATCH_SIZE = 5
PREDICT_PERIOD = 1
FIELD_LEN=len(FIELDS)
EPOCHS = 100
NEURONS = 4


def json2data(node):
  data=[]
  for k in FIELDS:
    data.append(node[k])
  return data

def getTimestamp(date):
  return dp.parse(date)

def moving_average(values, index, q):
  t = values[index-q+1, index+1]
  return np.average(t)

# load data
def loadData():
  data=[]
  with open('./data/{0}.json'.format(name)) as data_file:
    data = json.loads(data_file.read())
  
  col = FIELD_LEN * BLOCK_SIZE
  row = len(data)- BLOCK_SIZE - PREDICT_PERIOD + 1
  ub = len(data) - PREDICT_PERIOD - 1

  all_data= np.zeros((row, col+1))
  x_data = np.zeros((row, col))
  y_data = np.zeros(row)

  for i, d in enumerate(data):
    if i < BLOCK_SIZE-1:
      continue
    if i > ub:
      break
    idx = i-BLOCK_SIZE+1
    temp=[]
    for pre in range(BLOCK_SIZE-1, -1, -1):
      temp += json2data(data[i-pre])

    # x_data[i-BLOCK_SIZE] = temp
    all_data[idx, :col] = temp
    try:
      # y_data[i-BLOCK_SIZE] = data[i+PREDICT_PERIOD]['close']
      all_data[idx, col] = data[i+PREDICT_PERIOD]['close']
    except IndexError as e:
      print('i={0}, BLOCK_SIZE={1}, PREDICT_PERIOD={2}'.format(i, BLOCK_SIZE, PREDICT_PERIOD))

  all_data = np.asarray(all_data)
  # np.random.shuffle(all_data)
  
  [x_data, y_data] = np.split(all_data, [col], axis=1)

  TRAINING = int(len(x_data) * TRAIN_TEST)

  x_train_data = x_data[0:TRAINING, :]
  x_test_data = x_data[TRAINING:, :]
  mean = np.mean(x_train_data)
  x_train_data -= mean
  std = x_train_data.std()
  x_train_data /= std
  x_test_data -= mean
  x_test_data /= std
  y_train_data = y_data[0:TRAINING]
  y_test_data = y_data[TRAINING:]

  x_train_data= np.reshape(x_train_data, (x_train_data.shape[0], 1, x_train_data.shape[1]))
  x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], 1, x_test_data.shape[1]))
  # y_train_data = np.reshape(y_train_data, (y_train_data.shape[0], 1, 1))
  # y_test_data = np.reshape(y_test_data, (y_test_data.shape[0], 1, 1))

  return (x_train_data, y_train_data), (x_test_data, y_test_data)

from keras import models, layers, optimizers
def build_lstm(input_shape):
  model = models.Sequential()
  model.add(layers.LSTM(NEURONS, batch_input_shape=input_shape, name='lstm1', activation='tanh', use_bias=True, stateful=True, return_sequences=False))
  # model.add(layers.Dropout(0.2, name='drop1'))
  # model.add(layers.LSTM(400,return_sequences=False, name='lstm2', activation='tanh', use_bias=True, stateful=True))
  # model.add(layers.Dropout(0.2, name='drop2'))  
  model.add(layers.Dense(1, name='dense'))
  # model.add(layers.Activation('linear'))
  # rmsprop = optimizers.RMSprop(decay=0.0001)
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model

def train(x_train_data, y_train_data, x_test_data, y_test_data):
  x_train_data = x_train_data.reshape(x_train_data.shape[0], 1, x_train_data.shape[1])
  y_train_data = y_train_data.reshape(y_train_data.shape[0])
  split = int((len(x_train_data) - len(x_train_data)*0.05) / BATCH_SIZE) * BATCH_SIZE
  end = int(len(x_train_data) / BATCH_SIZE)*BATCH_SIZE
  x_val_data = x_train_data[split:end]
  y_val_data = y_train_data[split:end]
  x_train_data = x_train_data[:split]
  y_train_data = y_train_data[:split]

  x_test_data = x_test_data.reshape(x_test_data.shape[0], 1, x_test_data.shape[1])
  y_test_data = y_test_data.reshape(y_test_data.shape[0])
  end = int(len(x_test_data) / BATCH_SIZE)*BATCH_SIZE
  x_test_data = x_test_data[:end]
  y_test_data = y_test_data[:end]

  model = build_lstm((BATCH_SIZE, x_train_data.shape[1], x_train_data.shape[2]))
  # model = build_lstm((x_train_data[0].shape))
  verbose = 0
  if DEBUG >=2:
    model.summary()
    verbose = 1
  history = model.fit(x_train_data, y_train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=verbose, validation_data=(x_val_data, y_val_data))
  import utils
  if DEBUG >=3:
    utils.plot(history.history)

  prediction = model.predict(x_test_data, batch_size=BATCH_SIZE)

  log=False
  if DEBUG >=1:
    log=True  
  under, over, avg_err = utils.predict_result(prediction, y_test_data, log)
  # print('{0}>{1}, {2}<{3}, {4:.3f}<0.02'.format(under, len(prediction)/2, over, len(prediction)/5, abs(avg_err)))
  err = float(np.max(y_test_data))*0.006
  total = len(y_test_data)
  result={
    'under': under,
    'under_r': under*100/total,
    'over': over,
    'over_r': over*100/total,
    'avg_err': avg_err
  }
  if result['under_r'] > 60 and result['over_r'] < 20 and abs(avg_err) < err:
    model.save('{0}_lstm_b{1}p{2}_{3:.1f}_{4:.1f}_{5:.3f}.h5'.format(name, BATCH_SIZE, PREDICT_PERIOD, result['under_r'], result['over_r'], avg_err))
  if DEBUG >=3 :
    utils.plot_predict(prediction, y_test_data)
  return result

(x_train_data, y_train_data), (x_test_data, y_test_data) = loadData()
repeat = int(args['repeat'])
from colorama import Fore
for i in range(repeat):
  result = train(x_train_data, y_train_data, x_test_data, y_test_data)
  if result['avg_err'] < 0.5:
    color = Fore.GREEN
  else:
    color = Fore.WHITE  
  print('{0}train {1}: under={2:.2f}%({3}), over={4:.2f}%({5}), avg_err={6:.3f}'.format(color, i, result['under_r'], result['under'], result['over_r'], result['over'], result['avg_err']))  
print(Fore.WHITE)
# (x_train_data, y_train_data), (x_test_data, y_test_data) = loadData()
# from keras import models, layers
# model = build_rnn((x_train_data[0].shape))
# model.summary()
# history = model.fit(x_train_data, y_train_data, epochs=100, batch_size=128, validation_split=0.05, shuffle=True)
# import utils
# utils.plot(history.history)
# result = model.predict(x_test_data)
# utils.predict_result(result, y_test_data)
# utils.plot_predict(result, y_test_data)

