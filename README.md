# Operations_Department
Developing Deep Learning model to automate and optimize the disease detection processes at a hospital.
I will automate the process of detecting and classifying chest disease and reduce the cost and time of detection. 
There are 133 X-Ray chest images that belong to 4 classes: 
- Healthy 
- Covid-19
- Bacterial Pneumonia
- Viral Pneumonia 

## IMPORTING THE LIBRARIES AND DATASET

```PYTHON
import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

```
Looking at the directory

```python
os.listdir(XRay_Directory)
```
['2', '0', '3', '1']

```python

# Using image generator to generate tensor images data and normalize them
# Using 20% of the data for cross-validation  
image_generator = ImageDataGenerator(rescale = 1./255, validation_split= 0.2)

```
```python
train_generator = image_generator.flow_from_directory(batch_size = 40, directory= XRay_Directory, shuffle= True, target_size=(256,256), class_mode = 'categorical', subset="training")
```
Found 428 images belonging to 4 classes.


```python
validation_generator = image_generator.flow_from_directory(batch_size = 40, directory= XRay_Directory, shuffle= True, target_size=(256,256), class_mode = 'categorical', subset="validation")
```
Found 104 images belonging to 4 classes.

```python
train_images, train_labels = next(train_generator)
```
```python
train_images.shape
```
(40, 256, 256, 3)

```python
train_labels.shape
```
(40, 4)

```python
label_names = {0 : 'Covid-19', 1 : 'Normal' , 2: 'Viral Pneumonia', 3 : 'Bacterial Pneumonia'}
```

## VISUALIZING THE DATASET

```python
L = 6
W = 6

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis('off')

plt.subplots_adjust(wspace = 0.5) 
```
 <img src= "https://user-images.githubusercontent.com/66487971/91965499-39541b00-ed19-11ea-9210-e92a03702feb.png" width = 1000>

#  IMPORTING THE MODEL WITH PRETRAINED WEIGHTS

```python
basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256,256,3)))
```

```python
basemodel.summary()
```
 <img src= "https://user-images.githubusercontent.com/66487971/91966016-fe061c00-ed19-11ea-9872-e3b666c67690.png" width = 800>
 
 ```python
 for layer in basemodel.layers[:-10]:
  layers.trainable = False
  ```

## BUILDING AND TRAINING DEEP LEARNING MODEL

```python

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = "relu")(headmodel)
headmodel = Dropout(0.2)(headmodel)
headmodel = Dense(4, activation = 'softmax')(headmodel)

model = Model(inputs = basemodel.input, outputs = headmodel)

```

```python
model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.RMSprop(lr = 1e-4, decay = 1e-6), metrics= ["accuracy"])
```
  

```python
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

```
  ```python
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
```
  ```python
train_generator = image_generator.flow_from_directory(batch_size = 4, directory= XRay_Directory, shuffle= True, target_size=(256,256), class_mode= 'categorical', subset="training")
val_generator = image_generator.flow_from_directory(batch_size = 4, directory= XRay_Directory, shuffle= True, target_size=(256,256), class_mode= 'categorical', subset="validation")
```
Found 428 images belonging to 4 classes.
Found 104 images belonging to 4 classes.

  ```python
history = model.fit_generator(train_generator, steps_per_epoch= train_generator.n // 4, epochs = 10, validation_data= val_generator, validation_steps= val_generator.n // 4, callbacks=[checkpointer, earlystopping])
```

 <img src= "https://user-images.githubusercontent.com/66487971/91966725-ec714400-ed1a-11ea-8716-11b8e50f5519.png" width = 800>
 
 ## EVALUATING TRAINED DEEP LEARNING MODEL


  ```python
history.history.keys()
```
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
  ```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

plt.title('Model Loss and Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy and Loss')
plt.legend(['Training Accuracy', 'Training Loss'])
```
  ```python

```
  














































































