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
3. assets/handwriting-model-1.h5: the saved model.
4. 


### App
```
1. The app allows the user to select an input numer.
2. This number is used to lookup the ground truth name & its corresponding image from the csv test file.
3. The app uses tha trained model to predict the name from the given image.
4. Finally, the image, ground truth name and predicted name are displayed.
```

### Notes
- I followed methodologies from a couple of kaggle notebooks to derive the NN layers and the CTC loss function.
- The model is trained on approximately 330,000 images. (It look 12+ hrs on my mac, ran local cpu)
- I tried creating some test files with handwritten names (using mac img preview) & the model does not seem to predict them very well.
- It works much better on the test data.