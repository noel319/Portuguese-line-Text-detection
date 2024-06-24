# Portuguese-line-Text-dection

Install the dependencies:
 pip install -r requirements.txt

Run the main script:

python main.py images/sample.jpg

utils.py:

Contains the load_and_preprocess_image function that handles loading and preprocessing of the image, including resizing and normalizing.
main.py:

Loads the TrOCR processor and model.
Preprocesses the image using the function from utils.py.
Recognizes the text from the image using the TrOCR model.
Prints the recognized text.
