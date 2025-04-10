from django.shortcuts import render
from .models import Comment
from .forms import CommentForm
from django.http import JsonResponse
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import nltk
from nltk.corpus import stopwords
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import json
import os


MAX_LEN = 294 #used for padding sequences in training

nltk.download('stopwords')
stp = set(stopwords.words("english"))

with open("biGRU.pkl", 'rb') as file:
    biGRU = pickle.load(file)

with open("lstm2x.pkl", 'rb') as file:
    lstm2x = pickle.load(file)

with open("tokenizer.pkl", 'rb') as f:
    lstm_tokenizer = pickle.load(f)


def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    words = [word for word in text if word not in stp]
    text = " ".join(words)
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)  # use same maxlen as training
    return padded


def detectHate(content, mode):
    # ml model pkl loaded will come here and model.predict will be called.
    input = preprocess(content)
    if mode == 'lstm2x':
        prediction = lstm2x.predict(input)
        if prediction > 0.5:
            return True
        else:
            return False
    elif mode == 'biGRU':
        prediction = biGRU.predict(input)
        if prediction > 0.5:
            return True
        else:
            return False
    elif mode == 'distilBERT':
        return False


# Create your views here.
@require_GET
def model_info_view(request):
    model = request.GET.get("model")

    if model not in ["lstm2x", "biGRU", "model3"]:
        return JsonResponse({"error": "Invalid model name."}, status=400)

    base_url = f"/static/reports/{model}"
    response = {
        "report_img": f"{base_url}_report.png",
        "conf_matrix_img": f"{base_url}_cm.png",
    }
    return JsonResponse(response)


def home(request):
    form = CommentForm()
    comments = Comment.objects.filter(isHate=False)

    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data['content']
            mode = form.cleaned_data['mode']
            isHate = detectHate(content, mode)
            c = Comment(content=content, isHate=isHate)
            c.save()

            msg = "Hate Speech Detected, Content Blocked." if isHate else "Posted Successfully."
            return JsonResponse({"message": msg, "isHate": isHate})

    return render(request, 'index.html', {'form': form, 'comments': comments})
