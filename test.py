from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax


def predict_sentence_label(model_path, sentence):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1)
        print(probabilities)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return predictions.item()


user_sentence = input("请输入一个句子：")
predicted_label = predict_sentence_label("./saved_model", user_sentence)
print(f"Predicted label: {predicted_label}")
