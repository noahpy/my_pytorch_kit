
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from mnist.classifier.example import MyMnistModel
from my_pytorch_kit.model.classifier import ImageClassifier
import matplotlib.pyplot as plt


# 3. Create the Drawing Application
class DrawingApp:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        self.master.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="black")
        self.canvas.pack()

        self.label = tk.Label(self.master, text="Draw a digit", font=("Helvetica", 16))
        self.label.pack()

        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw_image.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit")

    def predict_digit(self):
        # Resize and process the image to fit the model's input requirements
        img = self.image.resize((28, 28))
        # img = ImageOps.invert(img)  # Invert colors to match MNIST

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img).unsqueeze(0)
        # plt.imshow(img_tensor[0].view(28, 28), cmap="gray")
        # plt.show()
        img_tensor = img_tensor.view(1, 1, 28, 28)

        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()
            self.label.config(text=f'Predicted Digit: {prediction}')


def interactive_test_set(model):
    mnist_test_dataset = MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    for i in range(10):
        model.use_softmax = True
        model.eval()
        x, y = mnist_test_dataset[i]
        x = x.view(x.shape[0], 784)
        y_pred = model.forward(x)
        model.use_softmax = False
        prediction = y_pred.argmax(dim=1, keepdim=True).item()
        plt.imshow(mnist_test_dataset[i][0].view(28, 28), cmap="gray")
        plt.title(f"Actual: {y}, Predicted: {prediction}")
        plt.show()


# 4. Run the Application
if __name__ == "__main__":
    hparams = {
        "feature_space": (64, 7, 7),
    }
    model = ImageClassifier(**hparams)
    model.load_model("models/classifier.pt")
    # interactive_test_set(model)
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()
