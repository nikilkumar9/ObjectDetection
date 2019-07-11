# Object Detection
**Basic Tensorflow SSD / RCNN Webcam Object Detection.**
https://trustmeimanengineer.de/en/object-detection/

Prerequisites:
* Python 3.6.8 - https://www.python.org/downloads/release/python-368/
* Git - https://git-scm.com/downloads
* Optionally (if you want to use the NVIDIA GPU of your computer):
	* Latest GPU driver - https://www.nvidia.com/Download/index.aspx?lang=en-us
	* CUDA 10.0 - https://developer.nvidia.com/cuda-10.0-download-archive
	* cuDNN - https://developer.nvidia.com/cudnn

1. Clone GermanEngineering/ObjectDetection Git repository.
* git clone https://github.com/GermanEngineering/ObjectDetection.git

2. Install dependencies
* If you can use a GPU:
	* pip install -r requirementsGpu.txt
* If you need to run on CPU:
	* pip install -r requirements.txt
	
3. Run program by executing:
* ObjectDetection.py


Depending on your specific usecase, you can pick another pre trained model with different speed and accuracy from the following selection:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
The accuracy of the model is ususally described by the mAP (mean Average Prescision) value on a range from 0 to 100 where higher numbers denote a higher accuracy of the model.
The speed of the model is usually given in ms for a computation with a specific system setup. Hence, you will probably not be able to get the same times, but can use it as a relative metric.
To use a different model, simply assign the name of the model as a string to the "MODEL_NAME" variable.

If you are looking for significantly faster detection speed you should take a look into YOLO detectors.
In general you are having the following options if you want to apply object detection:
1. Faster RCNN
Region Convolutional Neural Network
- Generates region proposals of where the objects in the image are probably located by grouping pixels with similar color, intensity, ...
- Region proposals are provided as input to the Convolutional Neural Network which outputs a feature map denoting the positions of specific characteristics in the image. 
- The last layer(s) of the network are used for classification by mapping the detected features to a class.
--> Comparably slow speed with high accuracies.

2. SSD
Single Shot Multi Box Detector
- Similar to RCNNs, but object localization and classification are done in one forward pass of the network.
--> Comparably higher speeds than RCNNs while maintaining good accuracies.

3. YOLO
You Only Look Once
(currently not available in the Tensorflow detection model zoo)
- Image is split into grid and multiple bounding boxes are created within each cell.
- Network outputs the probability values for each bounding box.
- All bounding boxes having a class probability above a certain threashold are used to classify and locate the object in the image.
--> Significantly faster but lower accuracies especially for small objects.

Main sources:
* Tensorflow on GitHub: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
* Harrison Kinsley - https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/


