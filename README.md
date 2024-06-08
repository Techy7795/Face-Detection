# Face Detection and Recognition Web Application

This Flask web application provides real-time face detection and recognition capabilities using a pre-trained InceptionResnetV1 model. Users can add new faces to the system by capturing images through their webcam and associating them with a username. The application then trains the model using the captured images to recognize the newly added faces.

## Features

- Real-time face detection using the MTCNN (Multi-task Cascaded Convolutional Networks) model.
- Face recognition based on pre-trained InceptionResnetV1 model.
- Ability to add new users and train the model with their faces.
- Web interface for interacting with the application.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/face-detection-recognition.git
cd face-detection-recognition
```

2. Run the Flask application:

```bash
python main.py
```

## Usage

1. Access the web interface by navigating to `http://localhost:5000` in your web browser.
2. Click on the "Add New User" button to add a new user.
3. Enter the username and the number of images to capture for training.
4. Follow the instructions to capture images of the new user's face using the webcam.
5. Once the training is complete, the model will be updated with the new user's information.
6. The application will then perform real-time face detection and recognition on the webcam feed, displaying the predicted identities.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and ensure that the code follows the style and conventions of the project.
3. Write tests for your code, if applicable.
4. Submit a pull request detailing your changes and the problem they solve.
