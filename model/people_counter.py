import torch
from pathlib import Path

class PeopleCounter:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.4  # Confidence threshold

    def predict(self, image):
        results = self.model(image)
        people = [det for det in results.pred[0] if det[-1] == 0]  # Class 0 is for 'person'
        return len(people), results.render()

    def train(self, train_data, val_data, epochs=10):
        self.model.train()
        self.model.train_data = train_data
        self.model.val_data = val_data
        self.model.train(epochs)

# Example usage
if __name__ == "__main__":
    model = PeopleCounter()
    # Add training and inference logic here
