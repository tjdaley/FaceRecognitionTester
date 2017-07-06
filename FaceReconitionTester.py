# FaceRecognitionTester.py
#
# Loads each of the frontal face recognition cascade files and runs them
# against a live video feed to see which works better in a particular
# environment.
#
# Copyright (c) 2017 by Thomas J. Daley, J.D.
# Author: Thomas J. Daley, J.D. <tjd@jdbot.us>
# URL: <http://www.jdbot.us>
# Distributed under the MIT license

import numpy as np
import cv2

class Classifier():
	"""
	Classifier - Utility class
	
	This is a simple class to keep everything relating to a single cascade classifier
	in one place. I started with a set of arrays, one for each property and that got
	cumbersome, so I added this class.
	"""
	
	def __init__(self, name, color, org, cascadeclassifier):
		"""
		Class constructor.
		
		Arguments:
			name - String. Name that is associated and displayed for this classifier.
			color - BGR tuble. Specifies the color that is associated with this classifier.
			org - (x, y) tuple. Origin for drawing the text label for this classifier
			cascadeclassifier - Instance of cv2.CasecadeClassifier
		"""
		self.name = name
		self.nameFormat = name + " {}"
		self.color = color
		self.org = org
		self.cascade = cascadeclassifier
		self.counter = 0
		
	def nameWithCounts(self):
		"""
		Returns a string with the name of the classifier and the number of faces recognized
		by this classifier so far.
		"""
		return self.nameFormat.format(self.counter)
		
	def nameWithPercentage(self, totalFrames):
		"""
		Returns a string with the name of the classifier and a percentage that is the number of
		faces recognized by this classifier divided by the total number of frames. When I'm testing
		the classifiers, I use them in an environment where there is only one face. That way, if
		a classifier sees more faces than frames, I know it's classifiying something as a face
		that is not a face (false positive).
		
		The usefulness of this percentage assumes that every frame contains a recognizable face.
		"""
		pad = " " * 100
		tmpName = (self.name + pad)[:20]
		tmpPct  = str(100*self.counter/totalFrames)
		parts = tmpPct.split(".")
		tmpWhole = ("000"+parts[0])[-3:]
		tmpDec = (parts[1]+"000")[:3]
		tmp = tmpName + tmpWhole + "." + tmpDec + " %"
		return tmp
		
	def incrementCounter(self):
		"""
		Increments a counter that tracks the number of faces recognized by this classifier.
		"""
		self.counter += 1

# Load cascade file for detecting faces		
# NOTE: You will need to download the cascade files yourself. They get updated now and then
# and you don't want the by-now old classifier cascade files that I used. Modify these lines
# to reflect the location of the cascade files on your system.
classifiers = []
classifiers.append(Classifier("Default", (255,0,0), (0,20), cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')))
classifiers.append(Classifier("Alt", (255,255,0), (0,50), cv2.CascadeClassifier('.\haarcascade_frontalface_alt.xml')))
classifiers.append(Classifier("Alt2", (0,255,0), (0,80), cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')))
classifiers.append(Classifier("AltTree", (0,0,255), (0,110), cv2.CascadeClassifier('.\haarcascade_frontalface_alt_tree.xml')))

# Total number of frames processed
frameCount = 0

# Connect to first (and probably only) camera
cap = cv2.VideoCapture(0)

# Continuously capture video frames until user presses 'Q'
while True:
	# Capture a frame
	ret, frame = cap.read()
	
	# Create a gray-scale version for use by the classifier
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #note reversed colors

	# Print classifier names in color that corresponds to their rectangles
	for classifier in classifiers:
		cv2.putText(frame, classifier.nameWithCounts(), classifier.org, cv2.FONT_HERSHEY_SIMPLEX, 1, classifier.color, 2)

	frameCount += 1
	
	# Run each classifier on the gray-scaled image
	for classifier in classifiers:
		
		# Look for faces
		faces = classifier.cascade.detectMultiScale(gray, 1.1, 5)

		# Draw a rectangle around each face
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), classifier.color, 2)
			classifier.incrementCounter()

	# Show the image. Loop until the user presses a key
	cv2.imshow('FACES', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# PRINT STATS
for classifier in classifiers:
	print(classifier.nameWithPercentage(frameCount))