import requests
import json
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=V5ZPRXSBI51DEJ75'
r = requests.get(url)
data = r.json()





fomratted_data = json.dumps(data, indent=2)

#ata_json = json.loads(fomratted_data)

#print(data_json["1"])

print(fomratted_data)