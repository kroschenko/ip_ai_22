import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Загрузка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Настройка предобученной MobileNet v3
base_model = MobileNetV3Small(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
# Разморозим некоторые верхние слои для тонкой настройки
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Добавление новых слоев для CIFAR-10
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Компиляция модели с уменьшенным learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Создаем обратный вызов для сохранения потерь на каждом шаге
class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

batch_loss_history = BatchLossHistory()

# Обучение модели с аугментацией данных
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=30,
                    validation_data=(x_test, y_test), callbacks=[batch_loss_history])

# Вывод финальной точности на обучающем наборе в процентах
final_accuracy = history.history['accuracy'][-1] * 100
print(f"Финальная точность на обучающем наборе: {final_accuracy:.2f}%")

# Построение графика потерь по батчам
plt.figure(figsize=(10, 5))
plt.plot(batch_loss_history.batch_losses, label='Train Loss per Batch', color='blue')
plt.title("Изменение ошибки по шагам в течение эпохи")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.show()
