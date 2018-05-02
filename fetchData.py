from twstock.stock import Stock
import json
import time
stock = Stock('0050', initial_fetch=False)

total = []
with open('tw0050.json', 'w') as data_file:
  for y in range(2004, 2019):
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
      with open('tw0050-{0}_{1}.json'.format(y, m), 'w') as perMonth:
        perMonth.write(json.dumps(output))
      data_file.write(json.dumps(total))
      time.sleep(5)
     
  

