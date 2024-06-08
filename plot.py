from flask import Flask, render_template, Response
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Function to generate frames (mocking video stream)
def gen_frames():
    # Mocking model performance data (accuracy and loss)
    acc_mobilenetv2 = [0.85, 0.88, 0.91, 0.93, 0.95]
    val_acc_mobilenetv2 = [0.82, 0.85, 0.87, 0.89, 0.90]
    loss_mobilenetv2 = [0.32, 0.28, 0.24, 0.21, 0.18]
    val_loss_mobilenetv2 = [0.40, 0.35, 0.32, 0.29, 0.27]

    acc_vgg16 = [0.80, 0.82, 0.85, 0.88, 0.90]
    val_acc_vgg16 = [0.75, 0.78, 0.80, 0.82, 0.84]
    loss_vgg16 = [0.40, 0.35, 0.30, 0.26, 0.22]
    val_loss_vgg16 = [0.50, 0.45, 0.40, 0.36, 0.33]

    acc_vgg19 = [0.78, 0.80, 0.82, 0.85, 0.87]
    val_acc_vgg19 = [0.72, 0.75, 0.78, 0.80, 0.82]
    loss_vgg19 = [0.42, 0.38, 0.34, 0.30, 0.26]
    val_loss_vgg19 = [0.52, 0.48, 0.44, 0.40, 0.37]

    epochs = range(1, len(acc_mobilenetv2) + 1)

    # Plotting accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc_mobilenetv2, label='MobileNet V2 Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_acc_mobilenetv2, label='MobileNet V2 Validation Accuracy', color='lightblue', marker='o')
    plt.plot(epochs, acc_vgg16, label='VGG16 Training Accuracy', color='red', marker='s')
    plt.plot(epochs, val_acc_vgg16, label='VGG16 Validation Accuracy', color='salmon', marker='s')
    plt.plot(epochs, acc_vgg19, label='VGG19 Training Accuracy', color='green', marker='^')
    plt.plot(epochs, val_acc_vgg19, label='VGG19 Validation Accuracy', color='lightgreen', marker='^')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # Plotting loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss_mobilenetv2, label='MobileNet V2 Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_loss_mobilenetv2, label='MobileNet V2 Validation Loss', color='lightblue', marker='o')
    plt.plot(epochs, loss_vgg16, label='VGG16 Training Loss', color='red', marker='s')
    plt.plot(epochs, val_loss_vgg16, label='VGG16 Validation Loss', color='salmon', marker='s')
    plt.plot(epochs, loss_vgg19, label='VGG19 Training Loss', color='green', marker='^')
    plt.plot(epochs, val_loss_vgg19, label='VGG19 Validation Loss', color='lightgreen', marker='^')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    # Saving the plot as image
    plt.savefig('performance_plot.png')

    # Reading the saved image
    img = cv2.imread('performance_plot.png')
    ret, buffer = cv2.imencode('.jpg', img)
    frame = buffer.tobytes()

    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
