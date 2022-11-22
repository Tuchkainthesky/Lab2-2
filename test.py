from keras.models import Model, model_from_json
from keras.preprocessing.image import image_utils
import numpy as np
import matplotlib.pyplot as plt
image_file_name = '3.png'
img = image_utils.load_img(image_file_name, target_size=(150, 150))
plt.imshow(img)
img_array = image_utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

json_file = open("model_json.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("saved_model.h5")
loaded_model.summary()

activation_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[3].output)
answer_model = Model(inputs=loaded_model.input, outputs = loaded_model.output)
activation_model.summary()
activation = activation_model.predict(img_array)
answer = answer_model.predict(img_array)
print(answer)
print(activation.shape)
plt.matshow(activation[0, :, :, 18], cmap='viridis')
plt.show()
images_per_row = 16
n_filters = activation.shape[-1]
size = activation.shape[1]
n_cols = n_filters // images_per_row
display_grid = np.zeros((n_cols * size, images_per_row * size))
for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = activation[0, :, :, col * images_per_row + row]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()

