# Find Me Out

## Project Description
"Find Me Out" is a face recognition system that uses a novel feature extractor and classifier to identify celebrities from a dataset of 400 images. This system achieves a high accuracy of 93%, making it robust and reliable for face recognition tasks.

## Key Features
- **High Accuracy**: Achieved 93% accuracy in recognizing faces using a Convolutional Neural Network (CNN).
- **Enhanced Preprocessing**: Utilized OpenCV for data augmentation, grayscale conversion, and resizing to ensure consistent input quality.
- **Advanced Training**: Leveraged TensorFlow with the Adam optimizer and Sparse Categorical Cross-Entropy loss function for effective multi-class face recognition.

## Installation
To run this project, you'll need to set up a Python environment with the required dependencies. Here's how you can do it:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/find_me_out.git
    cd find_me_out
    ```

## Usage
1. **Mount Google Drive**: Ensure your dataset is stored in Google Drive and mount it using the provided code snippet.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. **Data Loading and Preprocessing**: The system loads and preprocesses images for training.
    ```python
    import os
    import cv2
    import numpy as np
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    directory = "/content/drive/MyDrive/Colab Notebooks/Data/"
    celebrities = ["Scarlett_Johansson", "Elizabeth_Olsen", "Christian_Bale", "Chris_Evans"]

    data = []
    labels = []

    for i, celebrity in enumerate(celebrities):
        celebrity_dir = os.path.join(directory, celebrity)
        for h in range(1, 101):
            filename = f"IMG_{h}.jpg"
            if filename.endswith(".jpg"):
                file_path = os.path.join(celebrity_dir, filename)
                print(file_path)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (224, 224))  # Resize the image to a desired size
                data.append(image)
                labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25
    )

    train_data = x_train.astype("float32") / 255.0
    test_data = x_test.astype("float32") / 255.0

    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)
    ```
3. **Model Training**: Train the CNN model using the provided training code.
    ```python
    model = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 1)),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPool2D(),
            layers.Conv2D(24, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPool2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(4, activation="softmax"),
        ]
    )

    print(model.summary())

    # Compile and train the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )
    model.fit(train_data, y_train, batch_size=300, epochs=200, verbose=2)

    # Evaluate the model on the test set
    model.evaluate(test_data, y_test, batch_size=300, verbose=2)
    ```
4. **Prediction**: Use the trained model to predict and identify the celebrity in a new image.
    ```python
    from google.colab.patches import cv2_imshow
    test_image_path = (
        directory+"Chris_Evans/IMG_105.jpg"  # Replace with the path to your test image
    )
    test_image = cv2.imread(test_image_path)
    # cv2.imshow("Chris Evans",test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    test_image = cv2.resize(
        test_image, (224, 224)
    )  # Resize the image to match the model's input size
    test_image = np.expand_dims(test_image, axis=0)  # Add a batch dimension
    test_image = test_image.astype("float32") / 255.0  # Normalize pixel values

    # Make predictions on the test image
    predictions = model.predict(test_image)
    predicted_celebrity = celebrities[np.argmax(predictions[0])]
    cv2_imshow(cv2.imread(test_image_path))
    print("Predicted celebrity:", predicted_celebrity)
    ```
