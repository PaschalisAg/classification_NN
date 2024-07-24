import tensorflow as tf
from matplotlib import pyplot as plt
import os


def visualize_activations(model, validation_iterator, class_names, number_of_imgs, output_dir='plots/feature_maps/'):
    """
    Visualizes the activations of convolutional layers for a given model using a validation dataset.

    Args:
        model (tf.keras.Model): The trained Keras model.
        validation_iterator (tf.keras.preprocessing.image.Iterator): Iterator for the validation dataset.
        class_names (dict): Dictionary mapping class indices to class names.
        number_of_imgs (int): Number of images to visualize.
        output_dir (str, optional): Directory to save the feature map images. Defaults to 'plots/feature_maps/'.

    """
    os.makedirs(output_dir, exist_ok=True)

    # A Keras model that will output our previous model's activations for each convolutional layer
    activation_extractor = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers if "conv2d" in layer.name])

    def clean_plot(plot):
        """Remove axes from the plot."""
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)

    # Loads a sample batch of data
    sample_batch_input, sample_labels = next(validation_iterator)

    # Grabs the first `number_of_imgs` images
    sample_batch_input = sample_batch_input[:number_of_imgs]
    sample_labels = sample_labels[:number_of_imgs]

    # Makes predictions using model.predict(x)
    sample_predictions = model.predict(sample_batch_input)

    # Iterate over images, predictions, and true labels
    for i, (image, prediction, label) in enumerate(zip(sample_batch_input, sample_predictions, sample_labels)):
        image_name = f"Pneumonia_Type_{i}"

        # Gets predicted class with highest probability
        predicted_class = tf.argmax(prediction).numpy()

        # Gets correct label
        actual_class = tf.argmax(label).numpy()

        print(image_name)
        print(f"\tModel prediction: {prediction}")
        print(f"\tTrue label: {class_names[actual_class]} ({actual_class})")
        print(f"\tCorrect: {predicted_class == actual_class}")

        # Saves image file using matplotlib
        sample_image = image
        clean_plot(plt.imshow(sample_image))

        plt.title(f"{image_name}\nPredicted: {class_names[predicted_class]}, Actual: {class_names[actual_class]}")
        plt.show()
        plt.clf()

        # Get model layer output
        model_layer_output = activation_extractor(tf.expand_dims(sample_image, 0))

        # Iterate over each layer output
        for l_num, output_data in enumerate(model_layer_output):
            # Creates a subplot for each filter
            fig, axs = plt.subplots(1, output_data.shape[-1])

            # For each filter
            for i in range(output_data.shape[-1]):
                # Plots the filter's activations
                clean_plot(axs[i].imshow(output_data[0][:, :, i], cmap="gray"))

            plt.suptitle(f"{image_name} Conv {l_num}", y=0.6)
            plt.savefig(os.path.join(output_dir, f"{image_name}_Conv{l_num}.png"))
            plt.show()
            plt.clf()