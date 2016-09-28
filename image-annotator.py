import base64
import datetime
import hashlib
from timeit import default_timer as timer

from flask import Flask, render_template, request, jsonify
from gcloud import datastore
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

PROJECT_ID = 'image-annotator'
DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
IMAGE_LABEL_ENTITY_KIND = 'ImageLabel'
LABEL_ENTITY_KIND = 'Label'
credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials, discoveryServiceUrl=DISCOVERY_URL)
datastore_client = datastore.Client(PROJECT_ID)

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/label', methods=['POST'])
def label():
    file = request.files['file']
    if file and allowed_file(file.filename):
        file_contents = file.read()
        hash = sha512_hash(file_contents)

        cached_labels, cache_query_time_ms = query_cache(hash)
        if cached_labels:
            response = {
                'trace': {
                    'source': 'cache',
                    'cache_query_time_ms': cache_query_time_ms
                },
                'result': cached_labels['labels']
            }
            return jsonify(response)

        labels, api_query_time_ms = query_labels(b64_utf_8(file_contents))
        cache_save_time_ms = save_to_cache(hash, labels)

        response = {
            'trace': {
                'source': 'api',
                'cache_query_time_ms': cache_query_time_ms,
                'api_query_time_ms': api_query_time_ms,
                'cache_save_time_ms': cache_save_time_ms
            },
            'result': labels
        }
        return jsonify(response)


# Utils
def to_ms(seconds):
    return int(seconds * 1000)


def b64_utf_8(file_contents):
    return base64.b64encode(file_contents).decode('UTF-8')


def sha512_hash(file_contents):
    return hashlib.sha512(file_contents).hexdigest()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# Cloud Datastore Utils
def query_cache(hash):
    start = timer()
    key = datastore_client.key(IMAGE_LABEL_ENTITY_KIND, hash)
    entity = datastore_client.get(key)
    end = timer()
    return entity, to_ms(end - start)


def save_to_cache(hash, labels):
    key = datastore_client.key(IMAGE_LABEL_ENTITY_KIND, hash)
    datastore_entity = datastore.Entity(key)
    label_entities = [to_entity(label) for label in labels]
    datastore_entity.update({
        'labels': label_entities,
        'created': datetime.datetime.utcnow()
    })
    start = timer()
    datastore_client.put(datastore_entity)
    end = timer()
    return to_ms(end - start)


def to_entity(label):
    entity = datastore.Entity(datastore_client.key(LABEL_ENTITY_KIND))
    entity.update(label)
    return entity


# Vision API
def query_labels(image_content):
    service_request = service.images().annotate(body={
        'requests': [{
            'image': {
                'content': image_content
            },
            'features': [{
                'type': 'LABEL_DETECTION',
                'maxResults': 10
            }]
        }]
    })
    start = timer()
    service_response = service_request.execute()
    end = timer()
    labels = [{'label': r['description'], 'score': r['score']} for r in
              service_response['responses'][0]['labelAnnotations']]
    return labels, to_ms(end - start)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("8080"),
        debug=False
    )
