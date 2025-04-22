# draw_and_predict.py
"""
Interactive drawing interface for MNIST digit prediction using tkinter.
Lets user draw digits on a 28x28 grid and sends the result to a trained model.
"""

import tkinter as tk
import torch
import numpy as np
from model import load_trained_model
import scipy.ndimage

# Constants for the grid and UI layout
SIZE = 28         # Grid size
CELL = 20         # Size of each cell in pixels
CANVAS_SIZE = SIZE * CELL

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model(device=device)

# Drawing grid buffer (numpy array representing 28x28 pixel values)
grid = np.zeros((SIZE, SIZE), dtype=np.float32)

# Setup tkinter UI
root = tk.Tk()
root.title("Draw a Digit")
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.grid(row=0, column=0, columnspan=2)

# Initialize and draw grid rectangles
rectangles = [[None for _ in range(SIZE)] for _ in range(SIZE)]
for y in range(SIZE):
    for x in range(SIZE):
        rect = canvas.create_rectangle(
            x * CELL, y * CELL,
            (x + 1) * CELL, (y + 1) * CELL,
            fill="black", outline="gray"
        )
        rectangles[y][x] = rect

# Paint white when user clicks/drag over a grid cell
def paint_cell(x, y):
    if 0 <= x < SIZE and 0 <= y < SIZE:
        grid[y][x] = 1.0
        canvas.itemconfig(rectangles[y][x], fill="white")

# Get which cell was clicked based on pixel coordinates
def get_cell_coords(event):
    return event.x // CELL, event.y // CELL

def on_click(event):
    x, y = get_cell_coords(event)
    paint_cell(x, y)

def on_drag(event):
    x, y = get_cell_coords(event)
    paint_cell(x, y)

canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)

# Preprocess digit: blur and center mass
# Matches MNIST format more closely for better predictions
def preprocess(image):
    image = scipy.ndimage.gaussian_filter(image, sigma=0.9)  # smooth image
    cy, cx = scipy.ndimage.center_of_mass(image)             # get digit center
    shift_y = int(SIZE // 2 - cy)
    shift_x = int(SIZE // 2 - cx)
    image = scipy.ndimage.shift(image, shift=(shift_y, shift_x), order=1, mode='nearest')
    tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # shape [1,1,28,28]
    return tensor

# Send image to model and display prediction and confidence scores
def predict_digit():
    processed = preprocess(grid)
    with torch.no_grad():
        probs = model.predict_with_softmax(processed)
        pred = torch.argmax(probs, dim=1).item()
        result_label.config(text=f"Prediction: {pred}")

        # Display softmax probabilities
        prob_lines = []
        prob_list = probs.squeeze().cpu().tolist()
        for i, p in enumerate(prob_list):
            prob_lines.append(f"{i}: {p * 100:.2f}%")
        prob_text.config(text="\n".join(prob_lines))

# Reset the drawing grid to black
def clear_canvas():
    global grid
    grid = np.zeros((SIZE, SIZE), dtype=np.float32)
    for y in range(SIZE):
        for x in range(SIZE):
            canvas.itemconfig(rectangles[y][x], fill="black")
    result_label.config(text="")
    prob_text.config(text="")

# Buttons for user actions
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

# Labels to show prediction and class probabilities
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.grid(row=2, column=0, columnspan=2)

prob_text = tk.Label(root, text="", font=("Courier", 12), justify="left", anchor="w")
prob_text.grid(row=3, column=0, columnspan=2, padx=10, sticky="w")

root.mainloop()
