import json
import numpy as np
import dateutil.parser as dp

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", default=0, help="Show debug message 0:no message, 1:show predict, 2:show train+predict, 3: show plot")
ap.add_argument("-r", "--repeat", default=1, help="training times")
args = vars(ap.parse_args())

DEBUG = int(args['debug'])

code = '2317'
name = 'tw{0}'.format(code)

FIELDS = []
FIELDS.append('turnover')
FIELDS.append('high')
FIELDS.append('low')
FIELDS.append('open')
FIELDS.append('capacity')
FIELDS.append('close')

BLOCK_SIZE = 20
PREDICT_PERIOD = 1
FIELD_LEN=len(FIELDS)
TRAINING=6500

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
    d = dp.parse(data[i]['date'])
    # x_data[i-BLOCK_SIZE] = temp
    all_data[idx, :col] = temp
    try:
      # y_data[i-BLOCK_SIZE] = data[i+PREDICT_PERIOD]['close']
      all_data[idx, col] = data[i+PREDICT_PERIOD]['close']
    except IndexError as e:
      print('i={0}, BLOCK_SIZE={1}, PREDICT_PERIOD={2}'.format(i, BLOCK_SIZE, PREDICT_PERIOD))

  all_data = np.asarray(all_data)
  np.random.shuffle(all_data)
  
  [x_data, y_data] = np.split(all_data, [col], axis=1)

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

from keras import models, layers
def build_rnn(input_shape):
  model = models.Sequential()
  # model.add(layers.GRU(512, return_sequences=True, kernel_initializer='Orthogonal', name='gru1', input_shape=input_shape))
  model.add(layers.LSTM(300, input_shape=input_shape, return_sequences=True, name='lstm1'))
  model.add(layers.Dropout(0.2, name='drop1'))
  model.add(layers.LSTM(600,return_sequences=False, name='lstm2'))
  model.add(layers.Dropout(0.2, name='drop2'))
  model.add(layers.Dense(1, name='dense'))
  model.add(layers.Activation('linear'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model


def train():
  (x_train_data, y_train_data), (x_test_data, y_test_data) = loadData()
  model = build_rnn((x_train_data[0].shape))
  verbose = 0
  if DEBUG >=2:
    model.summary()
    verbose = 1
  history = model.fit(x_train_data, y_train_data, epochs=500, batch_size=128, validation_split=0.05, verbose=verbose)
  import utils
  if DEBUG >=3:
    utils.plot(history.history)

  result = model.predict(x_test_data)

  log=False
  if DEBUG >=1:
    log=True  
  under, over, avg_err = utils.predict_result(result, y_test_data, log)
  # print('{0}>{1}, {2}<{3}, {4:.3f}<0.02'.format(under, len(result)/2, over, len(result)/5, abs(avg_err)))
  err = float(np.max(y_test_data))*0.006
  if under > len(result)/2 and over < len(result)/5 and abs(avg_err) < err:
    model.save('{0}_rnn_b{1}p{2}_{3}_{4}_{5:.3f}.h5'.format(name, BLOCK_SIZE, PREDICT_PERIOD, under, over, avg_err))
  if DEBUG >=3 :
    utils.plot_predict(result, y_test_data)
  return under, over, avg_err

repeat = int(args['repeat'])
for i in range(repeat):
  under, over, avg_err = train()
  print('train {0}: under={1}, over={2}, avg_err={3:.3f}'.format(i, under, over, avg_err))


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

