# MNIST Digit Classifier

This project trains a neural network to classify handwritten digits using the MNIST dataset and includes a simple GUI where you can draw digits and get predictions.

## Files

- `train.py`: Trains the model and saves weights to `mnist_model.pth`
- `model.py`: Defines the model class and loading logic
- `draw_and_predict.py`: Tkinter interface to draw digits and predict them
- `mnist_experiment.py`: 
## Quick Start

Install dependencies:

```bash
pip install torch torchvision numpy scipy
```
Run Interactive Number Classification: 
```bash
python3 draw_and_predict.py
```
Train Model:
```bash
python3 train.py