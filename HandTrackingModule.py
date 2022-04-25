import cv2
import time
import mediapipe as mp

import utils


class HandDetector():

	def __init__(
			self, 
			mode = False, 
			max_hands = 2, 
			detection_confidence = 0.9, 
			tracking_confidence = 0.9
		) -> None:
		self.mode = mode
		self.max_hands = max_hands
		self.detection_confidence = detection_confidence
		self.tracking_confidence = tracking_confidence

		'''
		mp_hands is the file from where the Hands() class
		is retrieved
		'''
		self.mp_hands = mp.solutions.hands

		'''
		mp_draw is to get all the drawing utility functions
		and to draw on the hands
		'''
		self.mp_draw = mp.solutions.drawing_utils


		# Object for the Hands class
		self.hands = self.mp_hands.Hands()

		self.tip_ids = [4, 8, 12, 16, 20]

	def find_hands(self, frame, draw=True):
		# Process if there are hands in the image(frame)  
		# print(frame)
		self.results = self.hands.process(frame)

		'''
		If there are multiple hands in the image
		then draw on the image showcasing the hands
		'''
		if self.results.multi_hand_landmarks:
			for hand_landmarks in self.results.multi_hand_landmarks:
				if draw:
					self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

		return frame

	def find_positions(self, frame, hand_no = 0, draw = True):
		self.results = self.hands.process(frame)

		self.landmark_list = []

		if self.results.multi_hand_landmarks:
			my_hand = self.results.multi_hand_landmarks[hand_no]
			for id, lm in enumerate(my_hand.landmark):
				h, w, c = frame.shape
				cx, cy = int(lm.x * w), int(lm.y * h)

				self.landmark_list.append([id, cx, cy])

				if draw and id == 8:
					cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
				
		return self.landmark_list
    
	def fingersUp(self):
		fingers = []


		#Thumb
		if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
			fingers.append(1)
		else:
			fingers.append(0)

		# 4 Fingers
		for id in range(1, 5):
			if self.landmark_list[self.tip_ids[id]][2] > self.landmark_list[self.tip_ids[id] - 2][2]:
				fingers.append(1)
			else:
				fingers.append(0)

		return fingers


def main():
		
	prev_time, curr_time = 0, 0

	detector = HandDetector()

	'''
	Capturing the video from the camera.
	0 is for the camera of the machine
	'''
	capture = cv2.VideoCapture(0)
	# address: str = input("Enter the IP address given at the bottom of IP webcam: ")
	# capture.open(address + '/video')

	cv2.namedWindow("window_name", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("window_name", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	while True:

		# Capture the images from the video
		success, img = capture.read()

		# Convert the image from BGR to RGB
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		frame = utils.resize_to_full_screen(img)

		
		frame = detector.find_hands(frame)

		landmark_list = detector.find_positions(frame)

		if len(landmark_list) > 0:
			print(landmark_list)

		curr_time = time.time()
		fps = 1 / (curr_time - prev_time)
		prev_time = curr_time

		cv2.putText(frame, str(int(fps)), (30, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)


		cv2.imshow("Image", frame)

		# Stop the video when 'q' key is entered
		if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
  main()