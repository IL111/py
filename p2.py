import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import collections

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавление канала (для совместимости с Conv2D)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Проверка распределения меток в обучающем наборе
label_distribution = collections.Counter(y_train.argmax(axis=1))
print("Распределение меток в обучающем наборе:", label_distribution)

# Создание улучшенной модели
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)

# Оценка точности на обучающем и тестовом наборах
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность на обучающем наборе: {train_acc:.2f}")
print(f"Точность на тестовом наборе: {test_acc:.2f}")

# Визуализация потерь и точности
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Матрица ошибок
y_pred = model.predict(x_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='viridis')
plt.title('Матрица ошибок')
plt.show()

# Функция предсказания
def predict_image(image):
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return prediction.argmax()

# Тестирование случайного изображения из x_test
index = np.random.randint(0, len(x_test))  # Случайный индекс
test_image = x_test[index]  # Выбираем изображение
true_label = y_test[index].argmax()  # Истинная метка

# Визуализация изображения
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f"Тестовое изображение (истинное число: {true_label})")
plt.show()

# Предсказание
predicted_label = predict_image(test_image)
print(f"Предсказанное число: {predicted_label}")