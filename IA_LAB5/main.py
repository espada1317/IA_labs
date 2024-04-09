import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def detect_and_draw_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    (x, y, w, h) = faces[0]

    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    plt.show()


def blur_image(image, kernel_size=(5, 5)):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def sharpen_image(image, kernel_size=(3, 3), alpha=1.5, beta=-0.5):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, alpha * kernel + beta)
    return sharpened


# Load the image
image_path = 'test_images/35E54F.jpg'  # Replace 'path_to_your_image.jpg' with the actual path to your image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# Apply blur and sharpen filters
blurred_image = blur_image(image_rgb)
sharpened_image = sharpen_image(image_rgb)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_image)
plt.title('Blurred Image')
plt.axis('off')

plt.tight_layout()
plt.show()

detect_and_draw_face(image_path)
