# Face Recognition Tester

This is a simpley Python 3 script that creates classifiers from each of four different Haar cascade files. It then starts capturing a video stream and runs each frame through each classifier. At the top of the steaming video window, the name of each classifier and the number of faces that it has recognized is displayed. Once the user presses the 'q' key, the application stops and shows some performance statistics.

The performancde statistics aren't perfect. This application assumes there is one recognizable face in each frame. Therefore, each frame represents a positive test case.
If a classifier misses a lot of faces, then the total number of faces recognized by that classifier as a percentage of the total number of video frames processed will be less than 100.
If a classifier finds too many faces (false positives) it may "see" more than 100% of the faces visible. It's possible to have off-setting mistakes. This isn't QA for
the opencv frontal face classifiers. It's just a simple program to compare where each does well and where each does less well.

## Sample

```%> python FaceRecognitionTester.py```

## Author
Thomas J. Daley, J.D. <tjd@powerdaley.com> 