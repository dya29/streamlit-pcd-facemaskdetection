# import paket yang diperlukan
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# inisialisasi learning rate awal, jumlah epochs untuk pelatihan,
# dan ukuran batch
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# mengambil daftar gambar dalam direktori dataset, kemudian inisialisasi
# daftar data (mis., gambar) dan label kelas
print("[INFO] memuat gambar...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# melakukan one-hot encoding pada label
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# mengedit dataset
# membangun generator gambar pelatihan untuk augmentasi data
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# memuat jaringan MobileNetV2, memastikan layer FC di bagian atas
# tidak termasuk
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# membangun bagian atas model yang akan ditempatkan di atas
# base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# menempatkan model FC di bagian atas base model (ini akan menjadi
# model yang sebenarnya yang akan kita latih)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop melalui semua layer dalam base model dan membekukannya agar
# *tidak* diperbarui selama proses pelatihan pertama
for layer in baseModel.layers:
	layer.trainable = False

# mengompilasi model
print("[INFO] mengompilasi model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# melatih bagian atas jaringan
print("[INFO] pelatihan model...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# melakukan prediksi pada set data uji
print("[INFO] mengevaluasi jaringan...")
predIdxs = model.predict(testX, batch_size=BS)

# untuk setiap gambar dalam set data uji, kita perlu menemukan indeks 
# dari label dengan probabilitas prediksi terbesar yang sesuai
predIdxs = np.argmax(predIdxs, axis=1)

# menampilkan laporan klasifikasi dengan format yang rapi
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# menyimpan model yang telah dilatih ke dalam file
print("[INFO] menyimpan model deteksi masker...")
model.save("mask_detector.model", save_format="h5")

# plot loss dan akurasi pelatihan
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
