from PIL import Image
from predictor import predict

image = Image.open("segnale.jpg")

results, inference_time = predict(
    image,
    "ConvNeXt"
)

for classe, prob in results:
    print(f"{classe}: {prob:.2f}%")

print(f"Inference time: {inference_time*1000:.2f} ms")