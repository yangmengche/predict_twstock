import matplotlib.pyplot as plt
from colorama import Fore
import numpy as np

def plot(history):
  plt.subplot(211)
  plt.plot(range(1, len(history['loss'])+1), history['loss'], 'b', label='training loss')
  plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], 'r', label='val loss')

  plt.ylabel('loss')

  plt.subplot(212)
  plt.plot(range(1, len(history['mean_absolute_error'])+1), history['mean_absolute_error'], 'b', label='training error')
  plt.plot(range(1, len(history['val_mean_absolute_error'])+1), history['val_mean_absolute_error'], 'r', label='val error')
  plt.xlabel('Epochs')
  plt.ylabel('Validation MAE')

  plt.legend()
  plt.show()  

def plot_predict(predicts, truths):
  plt.plot(range(len(predicts[:,0])), predicts, 'b', label='predict')
  plt.plot(range(len(truths)), truths, 'r', label='truth')
  plt.ylabel('price')
  plt.legend()
  plt.show()

def predict_result(predicts, truths, log=False):
  under=0
  over=0
  total_diff=0
  truths = np.reshape(truths, len(truths))
  for i, r in enumerate(predicts):
    diff = truths[i] - r[0]
    total_diff += abs(diff)
    if abs(diff) < truths[i]*0.005:
      under += 1
      color = Fore.GREEN
    elif abs(diff) > truths[i]*0.01:
      color = Fore.RED
      over += 1
    else:
      color = Fore.WHITE
    if log:
      print('{0}+predict={1:.2f}, truth={2}, diff={3:.2f}'.format(color, r[0], truths[i], diff))

  avg_err = total_diff/len(truths)
  if log:
    print(Fore.WHITE+'under/over/total={0}/{1}/{2}, error={3:.2f}'.format(under, over, len(truths), avg_err))
  return under, over, avg_err