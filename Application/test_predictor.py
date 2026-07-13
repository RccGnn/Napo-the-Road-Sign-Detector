from PIL import Image
from predictor import predict

image = Image.open("segnale.jpg")

results = predict(
    image,
    "ConvNeXt" # Da provare senza StreamLit per eventuali aggiustamenti
)

for classe, prob in results:
    print(f"{classe}: {prob:.2f}%")