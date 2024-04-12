import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dlib
import csv


def sharpening_image(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, threshold)
    return sharpened


def blurring_image(image, kernel_size=(5, 5)):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def plot_initial_and_blured_image(img_path):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image = blurring_image(image_rgb)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Initial image')
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('Blurred image')
    plt.tight_layout()
    plt.show()


def extract_faces_from_image(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces


def get_face_recognition_coordinates(img_path):
    image = cv2.imread(img_path)
    detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        print("Photo does not contain faces.")
        return None

    plot_image_and_faces(image, faces)

    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    return x, y, w, h


def plot_image_and_faces(image, faces):
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


def check_color(image):
    return not (image[:, :, 0] == image[:, :, 1]).all() and not (image[:, :, 0] == image[:, :, 2]).all()


def check_orientation(image):
    height, width, _ = image.shape
    return height >= width


def check_eyes_level(face_landmarks, max_error=10):
    left_eye_y = face_landmarks.part(37).y
    right_eye_y = face_landmarks.part(46).y
    return abs(left_eye_y - right_eye_y) <= max_error


def check_head_size(face_coordinates, image):
    head_area = face_coordinates.area()
    image_area = image.shape[0] * image.shape[1]
    return 0.2 <= head_area / image_area <= 0.5


def passport_acceptance_system(img_path):
    image = cv2.imread(img_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) != 1:
        return False, "Photo is not accepted for passport. Photo does not contain exactly one person."

    face = faces[0]
    landmarks = predictor(gray_image, face)

    is_check_color = check_color(image)
    if is_check_color:
        is_check_orientation = check_orientation(image)
        if is_check_orientation:
            is_check_eyes_level = check_eyes_level(landmarks)
            if is_check_eyes_level:
                is_check_head_size = check_head_size(face, image)
                if is_check_head_size:
                    return True, "Photo is accepted for passport."
                else:
                    return False, ("Photo is not accepted for passport. The head area is not between 20% and 50% of "
                                   "photo.")
            else:
                return False, "Photo is not accepted for passport. Subject eyes are not on same level (max error 10 px)."
        else:
            return False, "Photo is not accepted for passport. Photo isn't in portrait orientation or square."
    else:
        return False, "Photo is not accepted for passport. Photo isn't colored."


def calculate_accuracy(csv_file):
    total_images = 0
    correctly_detected = 0

    print("Started calculating accuracy...")

    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            image_path = row['new_path']
            expected_label = True if row['label'] == 'True' else False

            is_accepted, response = passport_acceptance_system(image_path)

            print("Has been processed image # ", total_images + 1, " response: ", is_accepted, " - reason: ", response)

            total_images += 1
            if is_accepted == expected_label:
                correctly_detected += 1

    print("Finished calculating accuracy.")

    return correctly_detected / total_images if total_images > 0 else 0


if __name__ == '__main__':

    image_path = 'test_images/27DAC4.jpg'

    # TASK 1 - CODE
    plot_initial_and_blured_image(image_path)

    # TASK 2 - CODE
    face_coordinates = get_face_recognition_coordinates(image_path)
    if face_coordinates is not None:
        print("Face detected at coordinates:", face_coordinates)
    else:
        print("No face detected in the image.")

    # TASK 3 - CODE
    is_accepted, message = passport_acceptance_system(image_path)
    print("Image is accepted - ", is_accepted, ", reason - ", message)


    # TASK 4 - CODE
    csv_file_path = 'test.csv'
    accuracy = calculate_accuracy(csv_file_path)
    print(f"Accuracy: {accuracy * 100:.2f}%")
