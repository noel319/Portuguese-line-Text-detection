# transkribus.py
import requests
import time
import os

from config import USERNAME, PASSWORD, BASE_URL, COLLECTION_ID, MODEL_NAME, TRAINING_IMAGE_DIR, TRAINING_TRANSCRIPTION_DIR

def authenticate(username, password):
    url = f"{BASE_URL}/auth/login"
    data = {'user': username, 'pw': password}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()['sessionId']
    else:
        raise Exception("Authentication failed")

def upload_document(session_id, file_path, collection_id):
    url = f"{BASE_URL}/collections/{collection_id}/upload"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to upload document")

def request_handwritten_recognition(session_id, doc_id, page_ids):
    url = f"{BASE_URL}/jobs/{doc_id}/{page_ids}/textrecognition/htr"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to request handwritten recognition")

def get_recognition_results(session_id, doc_id):
    url = f"{BASE_URL}/collections/{COLLECTION_ID}/{doc_id}/fulldoc"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to retrieve recognition results")

def download_recognized_text(results, output_path):
    pages = results['md']['pages']
    full_text = ''
    for page in pages:
        full_text += page['tsList'][0]['ts']['content'] + '\n'
    
    with open(output_path, 'w') as file:
        file.write(full_text)

def wait_for_completion(session_id, job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            job_status = response.json()['state']
            if job_status == 'DONE':
                break
            elif job_status == 'FAILED':
                raise Exception("Recognition job failed")
            time.sleep(10)
        else:
            raise Exception("Failed to get job status")

def initiate_training(session_id, collection_id, model_name):
    url = f"{BASE_URL}/collections/{collection_id}/training"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    data = {
        'modelName': model_name,
        'type': 'HTR',  # Handwritten Text Recognition
        'baseLine': 'True'
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to initiate training")

def train_model(session_id, training_image_dir, training_transcription_dir, collection_id):
    # Upload training images
    image_files = sorted([os.path.join(training_image_dir, f) for f in os.listdir(training_image_dir) if f.endswith('.png')])
    for image_file in image_files:
        upload_document(session_id, image_file, collection_id)

    # Upload transcriptions
    transcription_files = sorted([os.path.join(training_transcription_dir, f) for f in os.listdir(training_transcription_dir) if f.endswith('.txt')])
    for transcription_file in transcription_files:
        upload_document(session_id, transcription_file, collection_id)

    # Initiate training
    training_response = initiate_training(session_id, collection_id, MODEL_NAME)
    return training_response

def request_text_detection(session_id, doc_id, page_ids):
    url = f"{BASE_URL}/jobs/{doc_id}/{page_ids}/textdetection"
    headers = {'Cookie': f"JSESSIONID={session_id}"}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to request text detection")
