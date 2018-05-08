import json
code = '2317'
name = 'tw{0}'.format(code)
data=[]
for y in range(1991, 2019):
  for m in range(1, 13):
    try:
      with open('./download_data/{0}-{1}_{2}.json'.format(name, y, m)) as data_file:
        json_data = data_file.read()
    except IOError:
      print('No file')
    else:
      data += json.loads(json_data)
with open('./data/{0}.txt'.format(name), 'w') as data_file:
  data_file.write(json.dumps(data))
