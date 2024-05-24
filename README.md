# Coffee-Bean-Image-Classification-Model
Coffee Bean Image Classification Project

---

# Image Classification Using Xception

This project demonstrates a deep learning approach for image classification using the Xception model, a state-of-the-art convolutional neural network (CNN). The model is trained and validated on a custom dataset, and a Gradio interface is provided for real-time predictions.

## Project Structure

- **Data Preparation**:
  - The dataset is loaded from specified directories and split into training, validation, and test sets.
  - Data augmentation techniques such as random flipping, rotation, and zooming are applied to enhance the training dataset.

- **Model Architecture**:
  - The base model used is Xception, pretrained on ImageNet, with the top layer removed to allow for custom classification.
  - Input images are resized to 224x224 pixels to match the expected input size of Xception.
  - Data augmentation layers are added to the input pipeline to improve model generalization.
  - The base model's output is connected to a global average pooling layer, followed by a dropout layer and a dense layer with a softmax activation function to output probabilities for each class.

- **Training**:
  - The base model layers are frozen to retain the pretrained weights.
  - The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
  - The training is performed for 10 epochs with real-time validation on the validation dataset.

- **Evaluation**:
  - Predictions are made on a batch of test images to evaluate the model performance.
  - The predicted class labels are compared to the true labels, and the images are displayed with the predicted class names.

- **Deployment**:
  - A Gradio interface is created to allow users to upload an image and receive predictions.
  - The interface shows the top three predicted classes with their respective probabilities.

## Code Explanation

### Data Loading and Augmentation
```python
batch_size = 32
img_size = (240, 240)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/Deep Learning A1/train/train',
    shuffle=True, batch_size=batch_size, image_size=img_size,
    validation_split=0.2, subset='training', seed=42
)
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/Deep Learning A1/test/test',
    shuffle=True, batch_size=batch_size, image_size=img_size,
    validation_split=0.2, subset='validation', seed=42
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/Deep Learning A1/test/test',
    shuffle=True, batch_size=batch_size, image_size=img_size
)
class_names = train_dataset.class_names
print(class_names)
```
- Loads the dataset from the specified directory.
- Splits the data into training, validation, and test sets.
- Applies shuffling and batching.

### Model Definition
```python
base_model = keras.applications.xception.Xception(input_shape=(224, 224, 3),
                                                  weights='imagenet',
                                                  include_top=False)
input = keras.layers.Input(shape=(240, 240, 3))
resized_input = keras.layers.Resizing(224, 224)(input)
preprocessed_input = keras.applications.xception.preprocess_input(resized_input)
flip = keras.layers.RandomFlip('horizontal')(preprocessed_input)
rotation = keras.layers.RandomRotation(0.2)(flip)
zoom = keras.layers.RandomZoom(0.2)(rotation)
base_model_output = base_model(zoom)
avg = keras.layers.GlobalAveragePooling2D()(base_model_output)
dropout = keras.layers.Dropout(0.2)(avg)
output = keras.layers.Dense(3, activation='softmax')(dropout)
model = keras.Model(inputs=input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False
```
- Defines the input shape and resizes the images.
- Applies data augmentation.
- Uses the Xception model as the base and adds custom layers for classification.

### Compilation and Training
```python
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
```
- Compiles the model with the Adam optimizer and a learning rate of 0.0001.
- Trains the model for 10 epochs with validation.

### Evaluation
```python
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
predictions = np.argmax(predictions, axis=-1)
print('Predictions:\n', predictions)
print('Labels:\n', label_batch)
```
- Evaluates the model on a batch of test images and prints the predictions and true labels.

### Visualization
```python
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
```
- Visualizes the predictions by displaying a batch of images with the predicted class labels.

### Gradio Interface
```python
def predict_image(img):
  img_4d = img.reshape(-1, 240, 240, 3)
  prediction = model.predict(img_4d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(3)}

image = gr.inputs.Image(shape=(240, 240))
label = gr.outputs.Label(num_top_classes=3)
iface = gr.Interface(fn=predict_image, inputs=image, outputs=label, interpretation='default')
iface.launch(debug=True, share=True)
```
- Defines a function to predict the class of an input image.
- Sets up a Gradio interface for real-time image classification.
