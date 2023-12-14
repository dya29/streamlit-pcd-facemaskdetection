import cv2
import mediapipe as mp
import tempfile
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np


#mediapipe inbuilt solutions 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model_path = "mask_detector.model"
model = load_model(model_path)

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

# Function to predict mask
def predict_mask(image):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return prediction

def main():

    #title 
    st.title('Face Mask Detection App')
    st.write('Please choice a detection method')

    st.subheader('Video Realtime')
    #creating a button for webcam
    use_webcam = st.button('Detect Mask')
    stframe = st.empty()
    
    st.subheader('Upload a File')
    #file uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    
    #temporary file name 
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    # membuatm model face detector dari disk serialized
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")

    if use_webcam:
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()  # Baca frame dari webcam

                # pastikan ret adalah True (pembacaan frame berhasil)
            if not ret:
                break
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ubah format frame menjadi RGB untuk Streamlit
            stframe.image(frame, channels="RGB")  # Menampilkan frame dengan deteksi masker di area yang telah disediakan

        vid.release()
        cv2.destroyAllWindows()

        st.success('Video is Processed')
        st.stop()
    else:
        st.write("")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=50)
        predictions = predict_mask(uploaded_file)
        mask = predictions[0][0]
        without_mask = predictions[0][1]

        # Membandingkan probabilitas untuk menentukan prediksi
        prediction_label = "Mask" if mask > without_mask else "Without Mask"
        prediction_confidence = max(mask, without_mask) * 100

        st.subheader(f"Predicted: {prediction_label}")
        st.subheader(f"Accuracy: {prediction_confidence:.2f}%")
if __name__ == '__main__':
    main()
