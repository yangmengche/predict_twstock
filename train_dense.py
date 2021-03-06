import json
import numpy as np
import dateutil.parser as dp

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", default=0, help="Show debug message 0:no message, 1:show predict, 2:show train+predict, 3: show plot")
ap.add_argument("-r", "--repeat", default=1, help="training times")
ap.add_argument("-s", "--save_config", action='store_true', help="save plot to file")
args = vars(ap.parse_args())

DEBUG = int(args['debug'])

code = '0050'
TRAIN_TEST = 0.8
# code = '2317'
# TRAINING=6000

name = 'tw{0}'.format(code)

FIELDS = []
# FIELDS.append('date')
# FIELDS.append('low')
# FIELDS.append('high')
FIELDS.append('open')
FIELDS.append('close')
# FIELDS.append('change')
# FIELDS.append('transaction')
# FIELDS.append('turnover')
# FIELDS.append('capacity')

BLOCK_SIZE = 5
BATCH_SIZE = 4
PREDICT_PERIOD = 1
FIELD_LEN=len(FIELDS)
EPOCHS = 100
NEURONS_1 = 128
NEURONS_2 = 16
SHUFFLE_ALL = True


def json2data(node):
  data=[]
  for k in FIELDS:
    data.append(node[k])
  return data

def getTimestamp(date):
  return dp.parse(date)

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
  if SHUFFLE_ALL:
    np.random.shuffle(all_data)

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

  y_test_data = np.reshape(y_test_data, len(y_test_data))

  return (x_train_data, y_train_data), (x_test_data, y_test_data)

from keras import models, layers, regularizers

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
config = tf.ConfigProto(inter_op_parallelism_threads = 4, intra_op_parallelism_threads = 4)
sess = tf.Session(config = config)
ktf.set_session(sess)

def build_dense(input_shape):
  model = models.Sequential()
  model.add(layers.Dense(NEURONS_1, activation='relu', input_shape=input_shape, bias_initializer='ones', kernel_regularizer = regularizers.l1(0.01)))
  if NEURONS_2 > 0:
    model.add(layers.Dense(NEURONS_2, activation='relu', bias_initializer='ones', kernel_regularizer = regularizers.l1(0.01)))
  model.add(layers.Dense(1))
  # model.add(layers.Activation('linear'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model

def train(model, x_train_data, y_train_data, x_test_data, y_test_data):
  verbose = 0
  if DEBUG >=2:
    model.summary()
    verbose = 1
  history = model.fit(x_train_data, y_train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, shuffle=True, verbose=verbose)
  import utils
  if DEBUG >=3:
    if args['save_config']:
      utils.plot(history.history, 'train.png')
    else:
      utils.plot(history.history)

  prediction = model.predict(x_test_data)

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
  if result['under_r'] > 57 and result['over_r'] < 23 and abs(avg_err) < err:
    model.save('{0}_dense_b{1}p{2}_{3:.1f}_{4:.1f}_{5:.3f}.h5'.format(name, BLOCK_SIZE, PREDICT_PERIOD, result['under_r'], result['over_r'], avg_err))
  if DEBUG >=3 :
    if(args['save_config']):
      utils.plot_predict(prediction, y_test_data, 'predict.png')
    else:
      utils.plot_predict(prediction, y_test_data)
  return result

(x_train_data, y_train_data), (x_test_data, y_test_data) = loadData()

# tune hyperparameter manually
model = build_dense((x_train_data.shape[1],))
repeat = int(args['repeat'])
from colorama import Fore
import time 
for i in range(repeat):
  t1 = time.time()
  result = train(model, x_train_data, y_train_data, x_test_data, y_test_data)
  t2 = time.time()
  if result['avg_err'] < 0.5:
    color = Fore.GREEN
  else:
    color = Fore.WHITE  
  print('{0}train {1}: under={2:.2f}%({3}), over={4:.2f}%({5}), avg_err={6:.3f}, time={7:.2f}'.format(color, i, result['under_r'], result['under'], result['over_r'], result['over'], result['avg_err'], t2-t1))
print(Fore.WHITE)

#tune by scikit-learn
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasRegressor
# model = KerasRegressor(build_fn=build_dense, verbose=0, input_shape=(x_train_data.shape[1],))
# batch_size = [4, 16, 32, 64, 128, 256]
# epochs = [50, 100, 200]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs = -1)
# grid_result = grid.fit(x_train_data, y_train_data)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))