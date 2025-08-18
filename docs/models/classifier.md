# Classifier

## Architecture
A series of convolutional layers act as a feature extractor, then an affine layer is to predict the class of the input image.

## Usage
The `ImageClassifier` class is implemented in the [`my_pytorch_kit.model.classifier`](../../my_pytorch_kit/model/classifier) module.
Import using:
```python
from my_pytorch_kit.models.classifier import ImageClassifier
```

Their application on the MNIST dataset can be found in the [`examples/mnist/classifier/classifier.py`](../../examples/mnist/classifier/classifier.py) module.
