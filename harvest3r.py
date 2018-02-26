#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import Counter
from datetime import datetime
import json
import logging
import os
import sys
import time

import requests

BASE_URL = 'http://epfl.elasticsearch.spinn3r.com'
LOG_DIR = '/usr/local/var/log'
DATA_DIR = '/Users/freened/dev/harvest3r'
BULK_SIZE = 20

HEADERS = {
  'X-vendor': 'epfl',
  'X-vendor-auth': 'J1Hr4Qc2a9UrU9tHweEO1KFDypA'
}

SWISS_NAMES = ['An Eilveis', 'CH', 'An Eilv\xe9is', 'Confederatio Helvetica', 'Confederation Suisse', 'Confederazione Svizzera', 'Confoederatio Helvetica', 'Conf\xe9d\xe9ration Suisse', 'Elvetia', 'Elve\u021bia', 'Helvetia', 'Isvicre', 'Iveits', 'Orileede switisilandi', 'Or\xedl\u1eb9\u0301\xe8de switi\u1e63ilandi', 'Schweiz', 'Schweizerische Eidgenossenschaft', 'Schweizi', 'Schwiz', 'Shvajcarija', 'Shvajcarska', 'Shvejcarija', 'Shvejcaryja', 'Soisa', 'Soissa', 'So\xefssa', 'Suica', 'Suis', 'Suisi', 'Suisilani', 'Suissa', 'Suisse', 'Suitza', 'Suiza', 'Suwiis', 'Suwisi', 'Suwizalan', 'Su\xedza', 'Su\xed\xe7a', 'Su\xefssa', 'Svajc', 'Svajcarska', 'Svajciarsko', 'Sveica', 'Sveicarija', 'Sveice', 'Sveis', 'Sveits', 'Sveitsi', 'Svejcaria', 'Svica', 'Svicarska', 'Sviss', 'Svisujo', 'Svizra', 'Svizzera', 'Svycarsko', 'Sv\xe1jc', 'Swetzaland', 'Swiiserlaand', 'Swise', 'Swiss', 'Swiss Confederation', 'Swis\u025b', 'Switizirandi', 'Switserland', 'Switzerland', 'Switzerland nutome', 'Szwajcaria', 'S\xfb\xeesi', 'Thuy Si', 'Th\u1ee5y S\u0129', 'Ubusuwisi', 'Uswisi', 'Y Swistir', 'Zvicer', 'Zvic\xebr', 'Zwitserland', 'i-Switzerland', 'isvecriya', 'isve\xe7riya', 'rui shi', 'sbijaralyanda', 'seuwiseu', 'shveitsaria', 'suisu', 'suisu lian bang', 'svijaralyanda', 'svisa', 'svitazaralainda', 'svitcarlantu', 'svitjarlend', 'svitjharlanda', 'svitjharlenda', 'swwyyz', 'swwyz', 'swys', 'swysra', 'swyys', '\u0128veits', '\u0130svi\xe7re', '\u0160vajcarska', '\u0160vaj\u010diarsko', '\u0160veica', '\u0160veicarija', '\u0160veice', '\u0160veits', '\u0160vica', '\u0160vicarska', '\u0160v\xfdcarsko', '\u0395\u03bb\u03b2\u03b5\u03c4\u03af\u03b1', '\u0428\u0432\u0430\u0458\u0446\u0430\u0440\u0438\u0458\u0430', '\u0428\u0432\u0430\u0458\u0446\u0430\u0440\u0441\u043a\u0430', '\u0428\u0432\u0435\u0439\u0446\u0430\u0440\u0438\u044f', '\u0428\u0432\u0435\u0439\u0446\u0430\u0440\u044b\u044f', '\u0428\u0432\u0435\u0439\u0446\u0430\u0440\u0456\u044f', '\u0547\u057e\u0565\u0575\u0581\u0561\u0580\u056b\u0561', '\u05e9\u05d5\u05d5\u05d9\u05d9\u05e5', '\u05e9\u05d5\u05d5\u05d9\u05e5', '\u0633\u0648\u0626\u0679\u0632\u0631 \u0644\u06cc\u0646\u0688', '\u0633\u0648\u0626\u06cc\u0633', '\u0633\u0648\u064a\u0633\u0631\u0627', '\u0633\u0648\u06cc\u0633', '\u0633\u0648\u06cc\u0633\u0631\u0627', '\u0938\u094d\u0935\u093f\u091c\u0930\u0932\u094d\u092f\u093e\u0923\u094d\u0921', '\u0938\u094d\u0935\u093f\u091f\u091c\u093c\u0930\u0932\u0948\u0902\u0921', '\u0938\u094d\u0935\u093f\u0924\u094d\u091d\u0930\u094d\u0932\u0902\u0921', '\u0938\u094d\u0935\u093f\u0938', '\u09b8\u09c1\u0987\u099c\u09b0\u09cd\u09b2\u09a3\u09cd\u09a1', '\u09b8\u09c1\u0987\u099c\u09be\u09b0\u09b2\u09cd\u09af\u09be\u09a8\u09cd\u09a1', '\u0ab8\u0acd\u0ab5\u0abf\u0a9f\u0acd\u0a9d\u0ab0\u0acd\u0ab2\u0ac5\u0aa8\u0acd\u0aa1', '\u0b38\u0b4d\u0b2c\u0b3f\u0b1c\u0b30\u0b32\u0b4d\u0b5f\u0b3e\u0b23\u0b4d\u0b21', '\u0bb8\u0bcd\u0bb5\u0bbf\u0b9f\u0bcd\u0b9a\u0bb0\u0bcd\u0bb2\u0bbe\u0ba8\u0bcd\u0ba4\u0bc1', '\u0c38\u0c4d\u0c35\u0c3f\u0c1f\u0c4d\u0c1c\u0c30\u0c4d\u0c32\u0c47\u0c02\u0c21\u0c4d', '\u0cb8\u0ccd\u0cb5\u0cbf\u0ca1\u0ccd\u0c9c\u0cb0\u0ccd\u200c\u0cb2\u0ccd\u0caf\u0cbe\u0c82\u0ca1\u0ccd', '\u0d38\u0d4d\u0d35\u0d3f\u0d31\u0d4d\u0d31\u0d4d\u0d38\u0d30\u0d4d\u200d\u0d32\u0d3e\u0d28\u0d4d\u200d\u0d21\u0d4d', '\u0dc3\u0dca\u0dc0\u0dd2\u0dc3\u0dca\u0da7\u0dbb\u0dca\u0dbd\u0db1\u0dca\u0dad\u0dba', '\u0e2a\u0e27\u0e34\u0e15\u0e40\u0e0b\u0e2d\u0e23\u0e4c\u0e41\u0e25\u0e19\u0e14\u0e4c', '\u0eaa\u0eb0\u0ea7\u0eb4\u0e94\u0ec0\u0e8a\u0eb5\u0ec1\u0ea5\u0e99', '\u0f66\u0f74\u0f60\u0f72\u0f4a\u0f0b\u0f5b\u0f62\u0f0b\u0f63\u0f7a\u0f53', '\u0f67\u0fb2\u0f74\u0f51\u0f0b\u0f67\u0fb2\u0f72\u0f0d', '\u1006\u103d\u1005\u103a\u1007\u101c\u1014\u103a', '\u10e8\u10d5\u10d4\u10d8\u10ea\u10d0\u10e0\u10d8\u10d0', '\u1235\u12ca\u12d8\u122d\u120b\u1295\u12f5', '\u179f\u17d2\u179c\u17b8\u179f', '\u30b9\u30a4\u30b9', '\u30b9\u30a4\u30b9\u9023\u90a6', '\u745e\u58eb', '\uc2a4\uc704\uc2a4']
CITY_NAMES = ['Geneva', 'Genf', 'Ginevra', 'Zurigo', 'Zermatt', 'Münchwilen', 'Porrentruy', 'Herisau', 'Schenkon', 'Payerne', 'Wengen', 'Lauterbrunnen', 'Schaffhausen', 'Solothurn', 'Einsiedeln', 'Gimel', 'Buchs', 'Flüelen', 'Gersau', 'Evolène', 'Lenzburg', 'Luzern', 'Lucerna', 'Lucerne', 'Raron', 'Winterthur', 'Domat', 'Romanshorn', 'Basel', 'Bâle', 'Basilea', 'Bellinzona', 'Appenzell', 'Bern', 'Berna', 'Amden', 'Dielsdorf', 'Aarau', 'Weinfelden', 'Willisau', 'Rheinfelden', 'Bad Zurzach', 'Samedan', 'Zürich', 'Saint-Maurice', 'Arlesheim', 'Zunzgen', 'Hinwil', 'Arth', 'Schwyz', 'Arbon', 'Saanen', 'Olten', 'Monthey', 'Novazzano', 'Bulle', 'Genève', 'St. Gallen', 'Sankt Gallen', 'San Gallo', 'Saint-Gall', 'Sursee', 'Stans', 'Liestal', 'Schleitheim', 'Pfaffikon', 'Lachen', 'Biel/Bienne', 'Biel', 'Bienne', 'Bienna', 'Zofingen', 'Ennetburgen', 'Marly', 'Fribourg', 'Küssnacht', 'Horgen', 'Bülach', 'Laufenburg', 'Zurich', 'Zuerich', 'Unterkulm', 'Suhr', 'Waldenburg', 'Lausanne', 'Losanna', 'Losanen', 'Neuchâtel', 'Pfäffikon', 'Grossandelfingen', 'Broc', 'Zug' 'Zugo', 'Lugano', 'La Sarraz', 'Pura', 'Laufen', 'Delemont', 'Kreuzlingen', 'Visp', 'Bremgarten', 'Wittnau', 'Andelfingen', 'Meilen', 'Schmerikon', 'Bière', 'Kussnacht', 'Tafers', 'Muri', 'Uster', 'Frauenfeld', 'Hochdorf', 'Aigle', 'Cevio', 'Delémont', 'Le Locle', 'Acquarossa', 'Munchwilen', 'Scuol', 'Emmetten', 'Brig', 'Altdorf', 'Schuepfheim', 'Neuchatel', 'Sissach', 'Sarnen', 'Conthey', 'Evolene', 'Sitten', 'Wangen an der Aare', 'Affoltern am Albis', 'Geneve', 'Bulach', 'Langnau', 'Ennetbürgen', 'Interlaken', 'Chur', 'Brugg', 'Schüpfheim', 'Flueelen', 'Renens', 'Leuk', 'Gränichen', 'Graenichen', 'Glarus', 'Baden', 'Frutigen', 'Steckborn', 'Thun', 'Thoune', 'Poschiavo']
OTHER = ['Matterhorn', 'Jungfrau', 'Eiger', 'Berner Oberland', 'Bernese Oberland', 'Grindelwald', 'Lavaux', 'Aare', 'Aar', 'Léman', 'Chillon', 'Gruyères', 'Rheinfall', 'Rhyfall', 'Chutes du Rhin', 'ETH', 'EPFL', 'Zurichsee', 'Genfersee']


class Harvester(object):
  """
  Abstract class containing the majority of logic. Harvesters for particular
  sources inherit after this class and should implement functions
  `construct_query` and `harvest_data`. This class should not be instantiated
  """

  def construct_query(self):
    raise NotImplementedError("This is an abstract class, use its derivations")

  def harvest_data(self, start_date, end_date):
    raise NotImplementedError("This is an abstract class, use its derivations")

  def response_is_valid(self, response):
    """ Check the status code of the response and log if it's not 200 """
    if response.status_code != 200:
      logging.error('Index request failed!')
      logging.error('Code: %d', response.status_code)
      logging.error('Content: %s', response.text)
      return False
    return True

  def log_failed_shards(self, data):
    """ Check if some shards failed and log it if it happens """
    if data['_shards']['failed'] > 0 :
      logging.warning('%d shards have failed and %d succeeded. Continuing.',
        data['_shards']['failed'],
        data['_shards']['successful']
      )

  def get_indices(self):
    """ Fetch the list of all indexes that contain 'content_' in the name """
    response = requests.get(BASE_URL + '/_aliases?ignore_unavailable',
                headers=HEADERS)

    if not self.response_is_valid(response):
      return None

    data = json.loads(response.text)
    return [k for k in data.keys() if 'content_' in k]

  def map_dates_to_indices(self, start_date, end_date):
    """ Select indices that contain data from a desired time period. """
    limits = dict()
    all_indices = self.get_indices()

    # We rely heavily on the naming schema of the indices.
    # The index parsing is intentionally very hardcoded, so that
    # if the names start to change, we should get an error ASAP
    for idx in all_indices:
      limits[idx] = dict()

      if idx[:8] == 'content_':
        year, month, day = map(int, (idx[8:12], idx[13:15], idx[16:18]))
        limits[idx]['start'] = datetime(year, month, day)
        limits[idx]['end'] = datetime(year, month, day)

      elif idx[:15] == 'merged_content_':
        year, month, day = map(int, (idx[15:19], idx[20:22], idx[23:25]))
        limits[idx]['start'] = datetime(year, month, day)

        year, month, day = map(int, (idx[29:33], idx[34:36], idx[37:39]))
        limits[idx]['end'] = datetime(year, month, day)

    # Filter the indices that might contain our data
    related_indices = []
    for index in all_indices:
      idx_start, idx_end = limits[index]['start'], limits[index]['end']
      if (start_date >= idx_start and start_date <= idx_end) or \
         (end_date >= idx_start and end_date <= idx_end) or \
         (start_date <= idx_start and end_date >= idx_end):
        related_indices.append(index)

    logging.info('Querying indices: ' + ', '.join(related_indices))

    return related_indices

  def persist_data(self, filename, batch, hits):
    """ Serialize the json as text data. """
    full_name = '{0}_{1}.json'.format(filename, batch)
    full_path = os.path.join(DATA_DIR, full_name)
    with open(full_path, 'a') as f:
      json.dump(hits, f, separators=(',', ':'))

  def download_data_from_period(self, start_date, end_date, consume_data, **kwargs):
    """ Get data from a limited period and dump it to a given directory """
    total_downloaded, batch = 0, 0
    bulk = kwargs.get('bulk', BULK_SIZE)

    # Get the indices
    indices = self.map_dates_to_indices(start_date, end_date)

    # Build the query
    query = self.construct_query(**kwargs)
    query['query']['bool']['filter'].append({
      "range" : {
        "published" : {
          "gte" : start_date.isoformat() + '||/d',
          "lte" : end_date.isoformat() + '||/d'
        }
      }
    })
    query['size'] = bulk

    # Send the first request
    url = '{0}/{1}/{2}'.format(BASE_URL, ','.join(indices), '_search?scroll=1m')
    print(url)
    logging.info('Sending an initial query to ' + url)

    print(query)
    response = requests.post(url, headers=HEADERS, json=query)

    if not self.response_is_valid(response):
      return None

    data = json.loads(response.text) # change to response.json()
    self.log_failed_shards(data)

    hits = data['hits']

    logging.info('Total records to fetch: %d, %d already downloaded.',
      hits['total'],
      len(hits['hits'])
    )
    total_downloaded += len(hits['hits'])

    print('Total is %d' % hits['total'])

    # Send subsequent requests until EOS
    while len(hits['hits']) > 0:
      consume_data(batch, hits['hits'])

      if len(hits['hits']) != bulk:
        break

      url = '{0}/{1}'.format(BASE_URL, '_search/scroll?scroll=1m')

      scroll_id = data['_scroll_id']
      response = requests.post(url, headers=HEADERS, data=scroll_id)
      data = json.loads(response.text) # change to response.json()

      print('Downloaded %d records' % total_downloaded)

      if self.response_is_valid(response):
        logging.info('Successfully downloaded a %d batch for %s - %s',
          len(data['hits']['hits']),
          start_date.strftime('%d/%m'),
          end_date.strftime('%d/%m')
        )
      else:
        return None

      self.log_failed_shards(data)
      hits = data['hits']

      total_downloaded += len(hits['hits'])
      batch += 1

      time.sleep(0.5) # delete me later on

    logging.info('Task finished. Downloaded total %d out of %d documents',
      total_downloaded, hits['total']
    )

    return total_downloaded

class TwitterHarvester(Harvester):
  """ A class for harvesting Twitter data from Spinn3r. """
  def __init__(self):
    super().__init__()

  def harvest_data(self, start_date, end_date):
    """ Download and persist the data from a given timespan. """
    data_filename = 'harvest3r_twitter_data_{0}_to_{1}'.format(
      start_date.strftime('%d-%m'),
      end_date.strftime('%d-%m')
    )

    def consume_data(batch_nr, hits):
      """ A wrapper function that also encapsulates the data filename """
      self.persist_data(data_filename, batch_nr, hits)

    return self.download_data_from_period(start_date, end_date, consume_data)

  def construct_query(self, **kwargs):
    """ Build a query that gets Twitter data from Switzerland. """
    query = {
      "query" : {
         "bool": {
           "must_not": [
            {"term": {"geo_country": "DE"}},
            {"term": {"geo_country": "FR"}},
            {"term": {"geo_country": "US"}},
            {"term": {"geo_country": "IT"}},
            {"term": {"geo_country": "AT"}},
            {"term": {"geo_country": "ES"}},
            {"term": {"geo_country": "GB"}},
            {"term": {"geo_country": "BE"}},
            {"term": {"geo_country": "LI"}}
          ],
          "filter": [
            {"term": {"domain": "twitter.com"}},
            {"match": {"source_title": "Daniel"}},
          ],
          "should": [{"term": {"geo_country": "CH"}}],
          "minimum_should_match": 1
         }
       }
    }

    mm_query = lambda name: {
      "multi_match" : {
        "query": name,
        "fields": ["geo_location^3", "source_location"]
      }
    }

    #for name in SWISS_NAMES + CITY_NAMES:
    #  query['query']['bool']['should'].append(mm_query(name))

    return query


def initialize_logger():
  """ Set the logger and its parameters. """
  today = datetime.now().strftime('%d-%m-%y')
  log_filename = 'harvest3r_' + today + '.log'
  log_filepath = os.path.join(LOG_DIR, log_filename)

  fh = logging.FileHandler(log_filepath)
  ch = logging.StreamHandler()
  logging.basicConfig(level=logging.INFO, handlers=[ch, fh])


if __name__ == '__main__':
  initialize_logger()

  # Check number of arguments
  if len(sys.argv) < 4:
    err_txt = 'Not enough arguments for the script. ' + \
          'Usage: ./harvest3r.py <source> <start-date> <end-date>'
    logging.error(err_txt)
    raise ValueError(err_txt)

  # Parse the dates passed as arguments
  try:
    start_date = datetime.strptime(sys.argv[2], '%d-%m-%Y')
    end_date = datetime.strptime(sys.argv[3], '%d-%m-%Y')
  except ValueError:
    logging.error('Could not parse the start/end dates: %s, %s',
      sys.argv[2], sys.argv[3]
    )
    raise

  assert start_date <= end_date

  # Check source
  harvesters = []
  if sys.argv[1] == 'twitter':
    harvesters = [TwitterHarvester()]
  elif sys.argv[1] == 'instagram':
    harvesters = [InstagramHarvester()]
  elif sys.argv[1] == 'news':
    harvesters = [NewsHarvester()]
  elif sys.argv[1] == 'all':
    harvesters = [TwitterHarvester(), InstagramHarvester(), NewsHarvester()]
  else:
    err_txt = 'Unknown source %s. Supported sources: twitter, instagram, news, all'
    logging.error(err_txt, sys.argv[1])
    raise ValueError(err_txt % sys.argv[1])

  # Here it goes!
  for harvester in harvesters:
    total_downloaded = harvester.harvest_data(start_date, end_date)

    if total_downloaded < 1:
      pass
