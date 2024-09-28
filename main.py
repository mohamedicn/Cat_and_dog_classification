import os
import shutil
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


print('1-file path')
print('2-one img')
print('chose service ?')
print('-----------')
user_input = input('::')
print('enter file or img path')
b = input('::')

cat_folder = 'F:/books/FOUR YEAR/Data Mining/New folder/project/cats/'
dog_folder = 'F:/books/FOUR YEAR/Data Mining/New folder/project/dogs/'
train_folder = 'F:/books/FOUR YEAR/Data Mining/New folder/project/train/'

# Create directories if not exist
os.makedirs(cat_folder, exist_ok=True)
os.makedirs(dog_folder, exist_ok=True)


base_model = MobileNetV2(weights='imagenet', include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  
predictions = Dense(2, activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = datagen.flow_from_directory(train_folder, 
                                                target_size=(224, 224), 
                                                batch_size=32, 
                                                class_mode='categorical', 
                                                subset='training')

validation_generator = datagen.flow_from_directory(train_folder, 
                                                    target_size=(224, 224), 
                                                    batch_size=32, 
                                                    class_mode='categorical', 
                                                    subset='validation')

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=5)

for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Function to classify and copy image to folders
def classify_and_copy_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]

    if class_idx == 0:  
        shutil.copy(img_path, cat_folder)
        print(f"Image {img_path} is classified as a Cat. Copied to Cats folder.")
    elif class_idx == 1: 
        shutil.copy(img_path, dog_folder)
        print(f"Image {img_path} is classified as a Dog. Copied to Dogs folder.")

# Function to classify and display the image
def classify_and_display_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    
    label = "Cat" if class_idx == 0 else "Dog"
    
    plt.imshow(image.load_img(img_path))  
    plt.title(f"This is a {label}") 
    plt.axis('off')
    plt.show() 


if int(user_input) == 1:
    for filename in os.listdir(b):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            file_path = os.path.join(b, filename)
            classify_and_copy_image(file_path, model)
elif int(user_input) == 2:
    classify_and_display_image(b, model)
else:
    print('Error: Invalid input')
