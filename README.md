# Object Detection
**Basic Tensorflow SSD / RCNN Object Detection application.**
https://trustmeimanengineer.de/en/object-detection/

Main sources:
* Tensorflow on GitHub: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
* Harrison Kinsley - https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/

Prerequisites:
* Python 3.6.8 from https://www.python.org/downloads/release/python-368/
* Git from https://git-scm.com/downloads
* Optionally (if you are using the NVIDIA GPU of your computer ):
* Latest GPU driver - https://www.nvidia.com/Download/index.aspx?lang=en-us
* CUDA 10.0 - https://developer.nvidia.com/cuda-10.0-download-archive
* cuDNN - https://developer.nvidia.com/cudnn
	
1. Clone GermanEngineering/ObjectDetection Git repository.
* Open command prompt
	* WINDOWS cmd ENTER
* Create new directory for Git projects.
	* mkdir GitProjects
* Navigate to created folder.
	* cd GitProjectscd DeepDream
* Clone repository.
	* git clone https://github.com/GermanEngineering/ObjectDetection.git

2. Install dependencies
* Navigate to folder containing the requirement files.
	* cd ObjectDetection
* List all files.
	* dir
* If you have a GPU you should execute:
	* pip install -r requirementsGPU.txt
* If you need to run on CPU execute:
	* pip install -r requirements.txt
* You can check all installed modules by executing:
	* pip list

3. Select your start and end image.
* Create new folders in the DeepDream/DreamImages directory for the start as well as the end image.
* Copy your images in the respective folder and rename them to "img_0.jpg"		
		
4. Configure the settings for your custom dream.
* Open the DeepDream.py file in a text editor.
* Change the dream1Folder and dream2Folder strings to the name of the folders you just created.
* Adapt other settings like the used layers, dream duration, fps, ... to your preferences (or leave them at default in case you're not sure)
	
5. Run the script by executing:
* DeepDream.py
