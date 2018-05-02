import json
tw0050=[]
for y in range(2004, 2019):
  for m in range(1, 13):
    try:
      with open('tw0050-{0}_{1}.json'.format(y, m)) as data_file:
        json_data = data_file.read()
    except IOError:
      print('No file')
    else:
      tw0050 += json.loads(json_data)
with open('tw0050.txt', 'w') as data_file:
  data_file.write(json.dumps(tw0050))
