from PIL import Image
from predictor import predict

# Test da terminale, senza app Streamlit, dei nostri modelli addestrati

# Un piccolo dataset da provare.
image = Image.open("images_examples/segnale2.jpg")

results, inference_time = predict(
    image,
    "ConvNeXt" # Inserire il nome del modello correttamente (case sensitive): EfficientNet o ConvNeXt
)

for classe, prob in results:
    print(f"{classe}: {prob:.2f}%")

print(f"Inference time: {inference_time*1000:.2f} ms")