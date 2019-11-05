import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import sys
import time
import mss
import math

# append path to object detection folder to be able to use its modules
sys.path.append(os.path.join(os.getcwd(), r"models\research\object_detection"))
sys.path.append(os.path.join(os.getcwd(), r"models\research"))
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
DOWNLOAD_FOLDER = "DownloadedModels/"
MODEL_NAME = "ssd_resnet_50_fpn_coco"
MODEL_FILE = DOWNLOAD_FOLDER + MODEL_NAME + ".tar.gz"

# Download Model
if not os.path.isfile(MODEL_FILE):
	opener = urllib.request.URLopener()
	opener.retrieve("http://download.tensorflow.org/models/object_detection/" + MODEL_FILE, MODEL_FILE)
	tarFile = tarfile.open(MODEL_FILE)
	for file in tarFile.getmembers():
		fileName = os.path.basename(file.name)
		if "frozen_inference_graph.pb" in fileName:
			tarFile.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.
# Actual model that is used for the object detection.
detectionGraph = tf.Graph()
with detectionGraph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.io.gfile.GFile(DOWNLOAD_FOLDER + MODEL_NAME + "/frozen_inference_graph.pb", "rb") as fid:
		serializedGraph = fid.read()
		od_graph_def.ParseFromString(serializedGraph)
		tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(os.path.join(os.getcwd(), r"models\research\object_detection\data\mscoco_label_map.pbtxt"))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Instanciate Tf session
with detectionGraph.as_default():
	sess = tf.compat.v1.Session(graph=detectionGraph)

# Run detection
def RunDetection(image):
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image, axis=0)
	image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	scores = detectionGraph.get_tensor_by_name('detection_scores:0')
	classes = detectionGraph.get_tensor_by_name('detection_classes:0')
	num_detections = detectionGraph.get_tensor_by_name('num_detections:0')

	# Actual detection.
	(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
														category_index, use_normalized_coordinates=True, line_thickness=9)

	return image


def DetectObjectsWebcam():
	"""Detect objects in webcam stream."""
	# Load webcam 
	# capture = cv2.VideoCapture(0)
	frame = cv2.imread('messi5.jpg', 0)
	result = RunDetection(frame)
	cv2.imshow('Object Detection', result)

	# while True:
	# 	# Read Webcam input
	# 	_, frame = capture.read()
	# 	frame = cv2.imread('C:\Users\nsenthilk2\Desktop\buses.jpg',0)
	# 	# Run detection
	# 	result = RunDetection(frame)
	# 	# Show result
	# 	cv2.imshow('Object Detection', result)
	# 	# Listen for abort
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		cv2.destroyAllWindows()
	# 		capture.release()
	# 		break


def DetectObjectsImages(inputFolder, outputFolder):
	"""Detect objects in all images of specified input folder and save new images with detection notations in output folder"""
	for path, subDirectories, files in os.walk(inputFolder):
		# create output path
		if not os.path.exists(path.replace(inputFolder, outputFolder)):
			os.makedirs(path.replace(inputFolder, outputFolder))

		for file in files:
			if file.lower().endswith(('.png', '.jpg', '.jpeg')):
				imagePath = os.path.join(path, file)

				# Load image
				image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

				# Resize image to largest dimension = 1280 px
				height, width = image.shape[:2]
				scaleFactor =  1280 / max(height, width)
				resizedImage = cv2.resize(image, (int(width * scaleFactor), int(height * scaleFactor)), interpolation=cv2.INTER_CUBIC)

				# Run object detection
				result = RunDetection(resizedImage)

				## Display images
				#cv2.imshow("ObjectDetection", result)
				#test = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
				#cv2.imshow("ObjectDetection2", test)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()

				# Save image
				cv2.imwrite(imagePath.replace(inputFolder, outputFolder), result)
				print("Processed image: {0}".format(os.path.basename(imagePath)))


def ObjectDetectionScreen():
	"""Detect objects on screen."""
	last_time = time.time()
	screenCapture = mss.mss()

	while True:
		frame = np.array(screenCapture.grab({"top": 40, "left": 0, "width": 800, "height": 640}))
		frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

		print("fps: {}".format(math.floor(1 / (time.time() - last_time))))
		last_time = time.time()

		proceddedImage = RunDetection(frame)
		cv2.imshow("Huhu =)", proceddedImage)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break


## Webcam Input
DetectObjectsWebcam()

#### Image Input
#inputFolder = os.path.join(os.getcwd(), r"Images\input")
#outputFolder = os.path.join(os.getcwd(), r"Images\output")
#DetectObjectsImages(inputFolder, outputFolder)

#### Screen Input
#ObjectDetectionScreen()
