import requests

# API endpoint
url = 'http://192.168.126.171:5000/predict'

# Path to the image file
image_path = r"C:\Users\Karthik Avinash\OneDrive\Desktop\6th Sem\Mini-project\1. Get Frames\frames\21bcs061\13.jpg"

# Read the image file
with open(image_path, 'rb') as file:
    # Prepare the POST request with the image file
    files = {'image': file}
    # Send the POST request
    response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    # If server returns an OK response, print the predicted labels
    data = response.json()
    predicted_labels = data['predicted_labels']
    print('Predicted Labels:', predicted_labels)
else:
    # If the server did not return a 200 OK response, print the error message
    print('Error:', response.text)
