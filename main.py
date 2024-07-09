# main.py
import os
from transkribus import authenticate, upload_document, request_handwritten_recognition, get_recognition_results, download_recognized_text, wait_for_completion, train_model, request_text_detection
from config import USERNAME, PASSWORD, COLLECTION_ID

def main():
    # Authenticate with Transkribus
    session_id = authenticate(USERNAME, PASSWORD)
    print(f"Authenticated successfully, session ID: {session_id}")

    # Define paths
    input_path = os.path.join('input', 'sample_document.png')
    output_path = os.path.join('output', 'recognized_text.txt')

    # Upload document
    upload_response = upload_document(session_id, input_path, COLLECTION_ID)
    print(f"Document uploaded successfully, response: {upload_response}")

    doc_id = upload_response['docId']
    page_ids = ','.join([str(page['pageId']) for page in upload_response['pages']])

    # Request text detection
    text_detection_response = request_text_detection(session_id, doc_id, page_ids)
    print(f"Text detection requested successfully, response: {text_detection_response}")

    job_id = text_detection_response['jobId']
    wait_for_completion(session_id, job_id)

    # Request handwritten recognition
    recognition_response = request_handwritten_recognition(session_id, doc_id, page_ids)
    print(f"Handwritten recognition requested successfully, response: {recognition_response}")

    job_id = recognition_response['jobId']
    wait_for_completion(session_id, job_id)

    # Retrieve recognition results
    results = get_recognition_results(session_id, doc_id)
    print(f"Recognition results retrieved successfully")

    # Save recognized text to output file
    download_recognized_text(results, output_path)
    print(f"Recognized text saved to {output_path}")

    # Train the model
    training_response = train_model(session_id, TRAINING_IMAGE_DIR, TRAINING_TRANSCRIPTION_DIR, COLLECTION_ID)
    print(f"Model training initiated successfully, response: {training_response}")

if __name__ == '__main__':
    main()
