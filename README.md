# BBCam
Blurred Background Camera

BBCam is a simple python project created to blur your background to maintain privacy.
This is made possible using Tensorflow BodyPix to split the foreground from the background.
Using OpenCV the mask is manipulated to be less obvious and give a seamless transition into the blurred background as seen in many commercial applications.
PyVirtualCam is used to send the openCV frames to existing virtual cameras such as OBS Virtual Webcam which can then be used by applications such as zoom, slack, Teams, etc.
