import requests
import json
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

port='http://localhost:8001/api/ner'
text="Next week, I am going to US. find  stripclub and bagel shop  around  Hotel Virginia Santa Barbara Tapestry Collection By Hilton in US."
request_example = {"sentence" : text}
test_get_response = requests.get(port, params=request_example, headers=headers)
print(test_get_response.json())
