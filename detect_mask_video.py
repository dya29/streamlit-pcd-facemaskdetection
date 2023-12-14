# import packages yang dibutuhkan
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# ambil dimensi frame dan kemudian membuat construct a blob dari dimensi tersebut
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# mwneruskan blob melalui jaringan dan mendapatkan face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inisialisasi daftar dari wajah, dari lokasi yang sesuai, dan daftar prediksi dari jaringn face mask
	faces = []
	locs = []
	preds = []

	# deteksi loop 
	for i in range(0, detections.shape[2]):
		# extract confidence (i.e., probability) yang terkait denga deteksi
		confidence = detections[0, 0, i, 2]

		# menyaringdeteksi yang lemah dengan memastikan confidence lebih besar dari confidence minimum
		if confidence > 0.5:
		# hitung koordinat (x, y)kotak pembatas objek
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# pastikan bounding boxes sesuai dengan dimensi frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract ROI face ,ubah dari urutan saluran BGR to RGB,ubah ukuran menjadi 224x224, dan proses terlebih dahulu
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# tambahkan face and bounding boxes ke respect lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# hanya membuat prediksi jika setidaknya satu wajah terdeteksi
	if len(faces) > 0:
		# untuk inference yang lebih cepat buatlah batch predictions pada *all*
		# wajah pada saat yang sama dan one-by-one predictions di dalam `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# mengembalikan 2-tuple pada face locations dan corresponding locations
	return (locs, preds)

# membuatm model face detector dari disk serialized
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# membuat model face mask detector dari disk
maskNet = load_model("mask_detector.model")

# deployment
# inisialisasi video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames dari video stream
while True:
	# ambil frame dari threaded video stream dan ubah ukurannya agar memiliki lebar maksimum 400 piksel
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect wajah dalam frame dan tentukan mana yang memakai masker mana yang tidak
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop lokasi wajah yang terdeteksi dan lokasi terkaitnya
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# tentukkan class label dan color yang akan kita gunakan membuat the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# masukkan the probability pada label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# menampilkan label dan bounding box rectangle pada output frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# tampilkan output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# key 'q' untuk break dari loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()