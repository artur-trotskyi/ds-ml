# models.py
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Завантаження моделі з файлу
@st.cache_resource
def load_model_from_file(model_name):
    model_path = os.path.join("models", model_name)
    return load_model(model_path)

# Завантаження історії навчання
@st.cache_data
def load_history(model_type):
    try:
        history_path = os.path.join("models", f"history_6{('_vgg16' if model_type == 'VGG16' else '')}.pkl")
        with open(history_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None