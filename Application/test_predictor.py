from PIL import Image
from predictor import predict

image = Image.open("images_examples/segnale2.jpg")

results, inference_time = predict(
    image,
    "EfficientNet"
)

for classe, prob in results:
    print(f"{classe}: {prob:.2f}%")

print(f"Inference time: {inference_time*1000:.2f} ms")