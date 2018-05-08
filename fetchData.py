from twstock.stock import Stock
import json
import time

code = '2317'
name = 'tw{0}'.format(code)
# stock = Stock('0050', initial_fetch=False)
stock = Stock(code, initial_fetch=False)

total = []
with open('./download_data/{0}.json'.format(name), 'w') as data_file:
  for y in range(1991, 2019):
    for m in range(1, 13):
      output=[]
      print('fetch: {0}/{1}'.format(y, m))
      try:
        stock.fetch(y, m)
      except IOError as e:
        print(e)
        data_file.write(json.dumps(output))

      for d in stock.data:
        data = {
          'date': d.date.isoformat(),
          'capacity': d.capacity,
          'turnover': d.turnover,
          'open': d.open,
          'high': d.high,
          'low': d.low,
          'close': d.close,
          'change': d.change,
          'transaction': d.transaction
        }
        output.append(data)
        total.append(data)
      with open('./download_data/{0}-{1}_{2}.json'.format(name, y, m), 'w') as perMonth:
        perMonth.write(json.dumps(output))
      data_file.write(json.dumps(total))
      time.sleep(5)
     
  

