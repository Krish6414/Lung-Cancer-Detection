import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Set directories
train_dir = "data/processed/ct_scans/train"
val_dir = "data/processed/ct_scans/valid"
test_dir = "data/processed/ct_scans/test"
model_save_path = "models/ct_scans/ct_vgg16_best.h5"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data generators with normalization
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_data = datagen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Load base VGG16 model (exclude fully connected layers)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint to save best model
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[checkpoint]
)

# Evaluate on test set
model.load_weights(model_save_path)
loss, accuracy = model.evaluate(test_data)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
