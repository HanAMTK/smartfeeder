# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils.video import VideoStream
import time


# define the paths to the Not Cat Keras deep learning model
MODEL_PATH = "cat_not_cat.model"
OBJ_LABEL = "Cat"

# initialize the total number of frames that *consecutively* contain
# cat along with threshold required to trigger the alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# initialize is the cat alarm has been triggered
FOUND = False

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
##time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# prepare the image to be classified by our deep learning network
	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

# classify the input image and initialize the label and
	# probability of the prediction
	(notCat, cat) = model.predict(image)[0]
	label = "Not Cat"
	proba = notCat

	# check to see if santa was detected using our convolutional
	# neural network
	if cat > notCat:
		# update the label and prediction probability
		label = "Cat"
		proba = cat

		# increment the total number of consecutive frames that
		# contain cat
		TOTAL_CONSEC += 1

		# check to see if we should raise the cat alarm
		if not FOUND and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that cat has been found
			FOUND = True

	# otherwise, reset the total number of consecutive frames and the
	# cat alarm
	else:
		TOTAL_CONSEC = 0
		FOUND = False

	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	frame = cv2.putText(frame, label, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()


