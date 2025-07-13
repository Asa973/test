import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# 1. Paramètres
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# 2. Prétraitement : Entraînement + Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'train-val-dataset/train-val-dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    'train-val-dataset/train-val-dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Prétraitement : Test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'test-dataset/test-dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# MODEL


# 1. Charger la base EfficientNetB0 pré-entraînée
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,         # on enlève la couche de classification d'origine
    input_shape=(224, 224, 3)
)

# 2. Geler les poids du modèle de base
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 3. Ajouter notre propre tête de classification
x = base_model.output
x = GlobalAveragePooling2D()(x)      # réduction des dimensions
x = Dropout(0.3)(x)                  # régularisation
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)  # 3 classes

# 4. Créer le modèle final
model = Model(inputs=base_model.input, outputs=output)

# 5. Compiler
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Afficher le résumé du modèle
model.summary()


# ... [4] Entraînement
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)


# ... [5] Courbes d'apprentissage
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()

plot_training(history)


# Évaluer le modèle
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
