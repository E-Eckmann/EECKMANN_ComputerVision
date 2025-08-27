# EECKMANN_ComputerVision
Computer Vision program that implements Face Detection, Template Matching, Optical Flow, and SIFT matching using OpenCV libraries and models 


Use the User Interface to start the program after running the .ipynb

Description:
This program provides several easily customizable Computer Vision features for images and videos. Results will be saved to the "OUTPUT" folder. Multiple features can be simultaneously applied onto a single image or video, or the features can be applied separately and individually. Video input can be either livestreamed or a previously saved video.

Face Detection
Uses pre-trained ML models to identify faces and classify the emotions and gender of people in an image/video. The faces on the image will be marked with rectangular bounding boxes, and the emotions and gender of the faces will be displayed above each box. The total amount of faces detected will be displayed in the top left corner. Be aware that this is currently the most computationally expensive of the features. Improvements will be made in the future.

How to use: Ensure that the "Face detection" button is set to True depending on whether you want to use images or videos. Drag images to the "INPUT" folder or videos to the "INPUT_VIDEO" folder.

Template Matching
Locates a smaller "template image" inside a larger image/video using your choice from a variety of OpenCV methods. The best matches will be marked with bounding boxes.

How to use: Ensure that the "Template matching" button is set to True depending on whether you want to use images or videos. Customize constant parameters to choose which OpenCV methods to use. Drag a single chosen template image to the "TEMPLATE" folder to compare it against each of the images in the "INPUT" folder or for each of the videos in the "INPUT_VIDEO" folder. Template must be smaller than compared images/frames or an Exception will be raised. If there are multiple template images in the "TEMPLATE" folder, only the first will be selected according to alphabetical order.

Optical Flow
Draws the optical flow detected in a sequence of images or the frames of a video.

How to use: Ensure that the "Optical flow" button is set to True depending on whether you want to use images or videos. Drag two or more images to the "INPUT" folder or videos to the "INPUT_VIDEO" folder.

SIFT (Scale Invariant Feature Transform)
Compares the SIFT keypoints between two images or between one images and a video. Lines will be drawn to show which keypoints match.

How to use: Ensure that the "SIFT matching" button is set to True depending on whether you want to use images or videos. Drag the image to be compared to the "SIFT" folder to compare it against each of the images in the "INPUT" folder or for each of the videos in the "INPUT_VIDEO" folder it. If there are multiple images in the "SIFT" folder, only the first will be selected according to alphabetical order.

Other considerations:
The "Customizable program constants" cell is where all constants that can be modified are found. This is where you can select which features to use or customize the parameters of the features. All constants are well documented with comments.

Multiple images or videos can be processed in a list. All images must be .jpg, .jpeg, or .png (but alpha channel not preserved) All images must be .mp4

Example input images will be found in the "data" folder.

The "utils" folder contains the defined functions used in the program.

The "trained_models" folder contains the pretrained face detection models sourced from OpenCV.

Credits:
Project programmer: Ethan Eckmann

Project mentor: Xandeep Alexander

Tutorials and code provided by OpenCV and others were crucial in creating this project.

https://github.com/opencv Face detection, emotion classification, and gender classification machine learning models sourced from OpenCV.

https://github.com/oarriaga/face_classification This is where some of the Face Detection code originated. Many modifications were made over the course of the project.

https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html Template Matching tutorial.

https://learnopencv.com/optical-flow-in-opencv/ Optical Flow tutorial.

https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html SIFT tutorial.

https://github.com/pinecone-io/examples/blob/master/learn/search/image/image-retrieval-ebook/bag-of-visual-words/bag-of-visual-words.ipynb SIFT bag of visual words / bag of features tutorial.

Known issues:
Text and bounding boxes may be difficult to see depending on your image/video dimensions. You must customize text and bounding box parameters manually.

Face detection does not work well with a series of images with varying dimsensions and varying distances of faces from the camera. The face detection parameters must be customized for the composition and dimensions of your chosen images.

The program will produce consistent face detection results for an image, but may produce a different set of consistent results for the same image that has been flipped horizontally (or otherwise edited). Flipping done in the Windows Photo app, any other photo editor, or using cv2.flip() function can change the results.
Initially having the photo flipped at all in photo editors causes different number of faces to be detected and emotion/gender classification to be different.
Initially having the photo flipped an odd number of times using cv2 functions causes emotion/gender classification to be different only.
Initially having the photo flipped an even number of times using cv2 functions changes nothing.
Differences in emotion, and gender accuracy due to orientation could be caused by the data each respective machine learning model was trained on.
However, the face detection for each image will always be processed in both its orignal state, AND its horizontally flipped state (using cv2) for the most consistent face detection results. The face detection results from both versions of the image are essentially "merged" together before emotion and gender classification. Despite this, the inconsistency problem with face detection still persists.
I can only speculate that this problem is not caused by any fault of this program but rather caused by metadata changes or differences in .jpg compression techniques between cv2 and various photo editors.

The emotion classifier struggles to identify disgust.

Face Detection with livestream video suffers from stuttering on each position update frame. This is due to the high computational cost of the face detection feature. Other features have stuttering as well, but it is most noticable here. I will work on fixing this flaw in the future.

Template matching often has low accuracy unless the template is found one-to-one inside an image. This is an inherent weakness of the simplistic template matching algorithim, and template matching was included in this program primarily to showcase the much more robust capabilities of SIFT matching.

Optical flow only tracks the best points initially detected on the first frame of a video. In the future, there will be an option to change this to instead have an update rate.

SIFT detection struggles when there is heavy noise such as camera blur.

Any form of SIFT detection with videos can sometimes create videos of very large resolution. This may cause the videos to be incompatable with windows media player. To get around this, you can use VLC media player instead.

Optical flow with images will raise an exception if the list of images provided are of varying dimensions.
