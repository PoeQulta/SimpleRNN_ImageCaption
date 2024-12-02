# Image Captioning with RNNs

This repository contains a PyTorch implementation of an image captioning model using Recurrent Neural Networks (RNNs).

## I. Getting the Dataset

*   Download the Flickr8k dataset from Kaggle using the provided link.
*   Unzip the downloaded file into a directory named 'data'.

## II. Dataset Preprocessing

This section covers the process of preparing the image and caption data for training the model. 

*   The code defines a function `clean_tokenize_caption` to clean and tokenize captions. It removes special characters, converts the caption to lowercase, and then splits the caption into individual words or tokens.

*   A vocabulary (`vocab`) is built from the captions using the `build_vocab` function. The vocabulary maps each unique word in the captions to a numerical index. Words occurring less than a specified threshold are excluded from the vocabulary. Special tokens are also added to the vocabulary.

*   The code converts captions to numerical sequences using `caption_to_sequence`. This function first cleans and tokenizes the caption using `clean_tokenize_caption`. Then, it adds special start and end tokens to the token list and converts each token to its corresponding index in the vocabulary. If a token is not present in the vocabulary, it is replaced with the index of the unknown token.

*   `ImageCaptionDataset` class loads and preprocesses the image and caption data. In the constructor, the image folder path and caption file path are initialized, and the captions are loaded from the file.
*   The `__getitem__` method loads an image and its corresponding caption, applies transformations to the image, and converts the caption to a numerical sequence using the built vocabulary.
*   The images are preprocessed using transformations such as resizing, cropping, conversion to tensors, and normalization.
*   A `DataLoader` is used for efficient batching and iteration of the dataset.
*   The `collate_fn` function is used in the data loaders to pad captions within each batch to the length of the longest caption in the batch.
*   The dataset is split into training, validation, and test sets.

## III. Model Definition

The code defines two main model classes: `FeatureExtract` and `CaptionModel`.

### **`FeatureExtract`**

*   Utilizes the Inception v3 model pre-trained on ImageNet for feature extraction.
*   Removes the fully connected layers of the Inception v3 model, using only the convolutional layers to extract image features.

### **`CaptionModel`**

*   Combines the extracted image features with an RNN to generate captions.
*   Image features are passed through a linear layer (`FeatureEmbedding`) to match the hidden size of the RNN.
*   Word embeddings are learned for each word in the vocabulary using an embedding layer (`embedding`).
*   An RNN (`rnn`) processes the word embeddings sequentially, generating hidden states that represent the context of the caption.
*   A fully connected layer (`fc`) maps the RNN's hidden state to a probability distribution over the vocabulary, predicting the next word in the caption.

## IV. Training

*   The model is trained on the training set and evaluated on the validation set.
*   The Adam optimizer is used to update the model's parameters during training.
*   The cross-entropy loss function is used to measure the difference between predicted and actual captions.
*   The model's performance is evaluated using the BLEU and CIDEr metrics.
*   The model is trained for a specified number of epochs, with the loss and evaluation metrics printed after each epoch.
*   The model's state is saved periodically during training.

## V. Evaluation

*   The trained model is evaluated on the test set using the BLEU and CIDEr metrics.
*   The generated captions and the ground truth captions for the test set are printed.

## VI. Additional Notes

*   The code utilizes GPUs for accelerated training if available.
*   The `tqdm` library is used to display progress bars during training and evaluation.
*   The `nltk` library is used for calculating the BLEU score.
*   The `pycocoevalcap` library is used for calculating the CIDEr score.
