# app.py
import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from models import load_model_from_file, load_history

# Налаштування сторінки
st.set_page_config(page_title="Класифікація одягу", layout="centered")

st.title("🧠 Класифікація зображень з Fashion MNIST")

# ===== Назви класів українською =====
class_names = [
    "Футболка/топ",  # 0
    "Штани",         # 1
    "Светр",         # 2
    "Сукня",         # 3
    "Пальто",        # 4
    "Босоніжки",     # 5
    "Сорочка",       # 6
    "Кросівки",      # 7
    "Сумка",         # 8
    "Черевики"       # 9
]

# ===== Інтерфейс =====
uploaded_file = st.file_uploader("📁 Завантажте зображення:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Відображення зображення
    st.image(image, caption="🖼️ Вхідне зображення", width=150)

    # Вибір моделі
    model_choice = st.radio("🔍 Оберіть модель для передбачення:", ("CNN Model", "VGG16 Model"))

    # Завантаження моделі та історії один раз при виборі
    if model_choice == "CNN Model":
        model = load_model_from_file("cnn_model_6.keras")
        history = load_history('CNN')
        processed_image = image.convert("L").resize((28, 28))
        image_array = np.array(processed_image).reshape(1, 28, 28, 1).astype("float32") / 255.0
    else:
        model = load_model_from_file("cnn_model_6_vgg16.keras")
        history = load_history('VGG16')
        processed_image = image.convert("RGB").resize((48, 48))
        image_array = np.array(processed_image).reshape(1, 48, 48, 3).astype("float32") / 255.0

    # Спінер + передбачення (тільки якщо ще не передбачили)
    if "prediction" not in st.session_state or uploaded_file.name != st.session_state.get("last_uploaded_file") or model_choice != st.session_state.get("last_model_choice"):
        with st.spinner("⏳ Модель аналізує зображення..."):
            time.sleep(1.5)  # штучна затримка
            prediction = model.predict(image_array)
            st.session_state.prediction = prediction
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.last_model_choice = model_choice
    else:
        prediction = st.session_state.prediction

    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[0][predicted_class]) * 100

    st.success("✅ Передбачення завершено!")
    st.markdown(f"### ✅ Передбачений клас: **{class_names[predicted_class]}** ({confidence:.2f}%)")

    st.subheader("📊 Ймовірності для кожного класу:")
    chart_data = {class_names[i]: float(prediction[0][i]) for i in range(10)}
    st.bar_chart(chart_data)

    # ===== Графіки навчання =====
    with st.expander("📉 Показати графіки функції втрат та точності моделі"):
        if history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history["loss"], label="Втрата (тренування)")
            ax1.plot(history["val_loss"], label="Втрата (валідація)")
            ax1.set_title("Функція втрат")
            ax1.legend()

            ax2.plot(history["accuracy"], label="Точність (тренування)")
            ax2.plot(history["val_accuracy"], label="Точність (валідація)")
            ax2.set_title("Точність")
            ax2.legend()

            st.pyplot(fig)
        else:
            st.warning("⚠️ Файл з історією навчання не знайдено.")
