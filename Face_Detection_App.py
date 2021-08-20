import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit begins

# Title
st.title("Face Detection App")

# Render the h1 block, contained in a frame of size 700x100.
components.html("<html><body><h3>Hello!! You can detect faces using this app.</h3></body></html>"
                , width=700, height=100)
st.markdown("<html><I>Make sure that your web camera is on.</I><br></html>",
            unsafe_allow_html=True)

# Building a sidebar
st.sidebar.subheader("Details of the person")
name = st.sidebar.text_input("Name of the Person 1")

st.write("Hi {}!!".format(name))
st.write("Welcome, hope you will enjoy using this app.")

st.markdown(f'<hr style="height:2px;border:none;color:#333;background-color:#333;" />', unsafe_allow_html=True)

st.write("Instruction: Enter 'q' to turn off the camera.")
st.info("if cv2.waitKey(1) & 0xFF == ord('q'): break")

st.header("Lets go!!")
if st.button("Can I detect your face ?"):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
