import cv2
import os
import argparse
import shutil
import urllib.request

# Standard argparse stuff
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-r', '--remote', type=bool, default=False,
                    help='use -r to change use the webcam (False) or the IP (True)')
parser.add_argument('-i', '--ip', type=str, default="http://ZZZ.ZZZ.ZZZ.ZZZ:8080/video",
                    help='use -ip to specify the URL of remote feed')

args = parser.parse_args()

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


def detection(gray, img):
    # Detect the faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Draw the rectangle around each face and a little circle on the center of detected face
    for (x_faces, y_faces, w_faces, h_faces) in faces:

        color = (255, 0, 0)
        stroke = 2

        # 2nd point for the rectangle around detected face (bottom right)
        end_cord_x = x_faces + w_faces
        end_cord_y = y_faces + h_faces

        # Coordinates of the center of detected face
        targ_cord_x = int((end_cord_x + x_faces)/2)
        targ_cord_y = int((end_cord_y + y_faces)/2)

        # Specific part of the frame for eye and smile detectors, using proper color code
        roi_gray = gray[y_faces:end_cord_y, x_faces:end_cord_x]
        roi_color = img[y_faces:end_cord_y, x_faces:end_cord_x]

        # Draw rectangle around detected face
        cv2.rectangle(img, (x_faces, y_faces),
                      (end_cord_x, end_cord_y), color, stroke)

        # Draw small circle on center of detected face
        cv2.circle(img, (targ_cord_x, targ_cord_y),
                   int(h_faces / 15), (0, 255, 0), 2)

        # Draw the rectangle around each eye
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.3, 18)
        for (x_eyes, y_eyes, w_eyes, h_eyes) in eyes:
            cv2.rectangle(roi_color, (x_eyes, y_eyes),
                          (x_eyes+w_eyes, y_eyes+h_eyes), (0, 180, 60), 2)

         # Draw the rectangle around each smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 18)
        for (x_smiles, y_smiles, w_smiles, h_smiles) in smiles:
            cv2.rectangle(roi_color, (x_smiles, y_smiles),
                          (x_smiles+w_smiles, y_smiles+h_smiles), (255, 0, 130), 2)
    return img


def main(args):
    # URL of the video feed
    URL = args.ip

    # A little flag that is used to determin if we're going to use a remote video feed
    IP = args.remote

    # Counter for the stills count
    img_counter = 0

    # Select proper VideoCapture device
    if(IP == True):
        cap = cv2.VideoCapture(URL)
    else:
        cap = cv2.VideoCapture(0)

    while(True):
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Start detection of desired features
        Feed = detection(gray, img)

        # Display
        cv2.imshow('Feed', Feed)

        # Check for pressed keys
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed, get a still
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, Feed)
            print("{} written!".format(img_name))
            img_counter += 1

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Call main function and pass args given by user
    main(args)