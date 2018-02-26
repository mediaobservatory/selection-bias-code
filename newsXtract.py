import os,json
import requests

BASE_URL  = 'http://epfl.elasticsearch.spinn3r.com/content*/_search'
BULK_SIZE = 100

SPINN3R_SECRET = os.environ['SPINN3R_SECRET']

HEADERS = {
  'X-vendor':      'epfl',
  'X-vendor-auth': SPINN3R_SECRET
}

query = {
   "size": BULK_SIZE,
   "query":{
      "bool":{
         "must":{
            "match":{
               "domain":"afp.com"
            }
         },
         "filter":{
            "range":{
               "published":{
                  "gte":"18/02/2017",
                  "lte":"20/02/2017",
                  "format":"dd/MM/yyyy"
               }
            }
         }
      }
   }
}


resp      = requests.post(BASE_URL, headers=HEADERS, json=query)
resp_json = json.loads(resp.text)

titles = set()

for r in resp_json['hits']['hits']:
  t = r['_source']['title']
  if t not in titles:
      print t
  titles.add(t)
