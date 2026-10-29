import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Лабораторная работа 2 - Вариант 3")
st.subheader("Гистограммы, контрастирование и высокочастотные фильтры")

uploaded_file = st.file_uploader(
    "Выберите изображение", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Исходное изображение",
        use_container_width=True,
    )

    section = st.selectbox(
        "Выберите блок методов",
        [
            "Построение, эквализация гистограммы и линейное контрастирование",
            "Высокочастотные фильтры (увеличение резкости)",
        ],
    )

    if (
        section
        == "Построение, эквализация гистограммы и линейное контрастирование"
    ):
      method = st.selectbox(
          "Метод", ["Эквализация гистограммы", "Линейное контрастирование"]
      )

      if method == "Эквализация гистограммы":
        processed = cv2.equalizeHist(gray)
      else:
        min_v = float(np.min(gray))
        max_v = float(np.max(gray))
        if max_v > min_v:
          processed = np.clip(
              (gray.astype(np.float32) - min_v) * (255.0 / (max_v - min_v)),
              0,
              255,
          ).astype(np.uint8)
        else:
          processed = gray

      st.image(
          cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB),
          caption=method,
          use_container_width=True,
      )

      st.subheader("Сравнение гистограмм")
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

      ax1.hist(gray.ravel(), 256, [0, 256], color="black", alpha=0.7)
      ax1.set_title("Исходная гистограмма")
      ax1.set_xlim([0, 256])

      ax2.hist(processed.ravel(), 256, [0, 256], color="royalblue", alpha=0.7)
      ax2.set_title(f"Гистограмма: {method}")
      ax2.set_xlim([0, 256])

      st.pyplot(fig)

    else:
      filter_name = st.selectbox(
          "Фильтр", ["Фильтр Лапласа", "Нерезкое маскирование", "Фильтр Собеля"]
      )

      if filter_name == "Фильтр Лапласа":
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        res = cv2.convertScaleAbs(
            cv2.addWeighted(gray.astype(np.float32), 1.0, lap, -1.0, 0)
        )
      elif filter_name == "Нерезкое маскирование":
        blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        res = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
      else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        res = cv2.convertScaleAbs(
            cv2.addWeighted(gray.astype(np.float32), 1.0, mag, 0.3, 0)
        )

      st.image(
          cv2.cvtColor(res, cv2.COLOR_GRAY2RGB),
          caption=filter_name,
          use_container_width=True,
      )
