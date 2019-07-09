import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import sys

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

# Load webcam 
capture = cv2.VideoCapture(0)

# Run detection
with detectionGraph.as_default():
  with tf.compat.v1.Session(graph=detectionGraph) as sess:
    while True:
      # Read Webcam input
      _, frame = capture.read()

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(frame, axis=0)
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
      vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                            category_index, use_normalized_coordinates=True, line_thickness=8)

      # Show result
      cv2.imshow('Object Detection', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        capture.release()
        break
