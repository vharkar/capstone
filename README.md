### Capstone Project - Handwriting Recognition using CRNN in Keras

* This project is based on data and notebooks in [Kaggle](https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/data)
* The project uses a CNN + RNN model for handwriting recognition
* The model is trained with CTC loss.

### Reading material

* [Understanding CTC](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c) 
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
* [Spatial filters in image processing](https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-spatial-filters-convolution/)

### Project Structure
1. notebook/handwriting.ipynb : Data Exploration, model creation, training & save.
2. assets/written_name_test_v2.csv : the mapping of names to images
3. assets/created_names.csv : the mapping of names to images that I created
4. assets/handwriting-model-1.h5: the saved model.
5. images/test - has the test images.
6. my_images - has images I generated
7. tmp_images - a tmp dir to place converted images.
8. app-new.py - final version of app
9. CapstonePresentation.key - a brief presentation.


### App
```
1. The app (app-new.py) has 4 tabs.
2. Tab 1 
  * User selects a number (the range is the number of test files checked in to git = 100).
  * This number is used to lookup the ground truth name & its corresponding image from the csv test file.
  * The app predicts the name from the image with the trained model.
  * Finally, the image, ground truth name and predicted name are displayed.
3. Tab 2
  * A canvas to take in handwritten text and predict the name.
4. Tab 3
  * Names predicted from img files (9 total) created with img preview (same methodology as Tab 1)
5 Tab 4
  * Names predicted from text -> converted to handwriting type image.  
```

### Notes
- I followed methodologies from a couple of kaggle notebooks to derive the NN layers and the CTC loss function.
- The model is trained on approximately 330,000 images. (It look 12+ hrs on my mac, ran local cpu)
- The model works well on the test data.
- I tried the following strategies to predict names from user created images:
  - predict from canvas image 
  - creating image files with handwritten names (using mac img preview)
  - convert a text name to handwriting image with ImgKit
- In either of the above, the model does not seem to predict them very well.
