import fasttext
import os

dataset_file = "data.txt"
if not os.path.exists(dataset_file):
    print(f"Dataset file '{dataset_file}' tidak ditemukan")
    exit()

print("Melatih model FastText...")
model = fasttext.train_supervised(input=dataset_file, lr=0.5, epoch=10, wordNgrams=2)

model_filename = "text_classification_model.ftz"
model.save_model(model_filename)
print(f"Model berhasil disimpan ke '{model_filename}'")

print("\nMenguji model...")
test_sentences = [
    "I love the quality of this product",
    "The delivery was terrible",
    "This is the best service I have ever experienced",
    "Not worth the money",
    "konten sampah",
    "streaming sampah, tapi saya suka",
    "Judi online bikin hidup hancur",
    "Saya suka judi online"
]

for sentence in test_sentences:
    label, confidence = model.predict(sentence)
    print(f"Kalimat: '{sentence}'")
    print(f"Prediksi: {label[0]} (Confidence: {confidence[0]:.2f})")
    print("-" * 50)
