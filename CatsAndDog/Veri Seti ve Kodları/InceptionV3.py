import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Dosya yollarını tanımlama
base_dir = r'C:\Users\izbur\Desktop\KediKopek'
train_dir = os.path.join(base_dir, 'egitim')
test_dir = os.path.join(base_dir, 'test')

# Veri kümesini yükleme ve etiketleme
def load_data(directory):
    images = []
    labels = []
    for label, category in enumerate(['Kedi', 'Kopek']):
        path = os.path.join(directory, category)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img = img.resize((299, 299))  # Gerekirse görüntüyü yeniden boyutlandırma
            img = np.array(img)
            if img.shape == (299, 299, 3):
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Veri kümesini yükleme
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# Veri kümesini train ve validation olarak ayırma
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Derin öğrenme modelini tanımlama ve önceden eğitilmiş ağı yüklemek
def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Modeli oluşturma
model = create_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim ve değerlendirme
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val))

# Değerlendirme ölçütlerini hesaplama
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Sonuçları yazdırma
print("Sonuçlar:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
