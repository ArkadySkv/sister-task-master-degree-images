import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

img = Image.open('./images/glioma1.jpg')
img_data = np.array(img)
width, height = img.size

red_channel = img_data[:,:,0]
green_channel = img_data[:,:,1]
blue_channel = img_data[:,:,2]

red_channel -= 50
green_channel -= 50
blue_channel -= 50

modified_img_data = np.zeros_like(img_data)
modified_img_data[:,:,0] = red_channel
modified_img_data[:,:,1] = green_channel
modified_img_data[:,:,2] = blue_channel

modified_img = Image.fromarray(modified_img_data)

histogram_red = np.histogram(red_channel.flatten(), bins=256, range=(0, 256), density=True)[0]
histogram_green = np.histogram(green_channel.flatten(), bins=256, range=(0, 256), density=True)[0]
histogram_blue = np.histogram(blue_channel.flatten(), bins=256, range=(0, 256), density=True)[0]

entropy_red = -np.sum(histogram_red * np.log2(histogram_red + 1e-10))
entropy_blue = -np.sum(histogram_blue * np.log2(histogram_blue + 1e-10))

if entropy_red < entropy_blue:
    sys.stdout.write("A red channel has more low entropy")
else:
    sys.stdout.write("A blue channel has more low entropy")

def logarithmic_transformation(pixel_value, c=1):
    return int(c * (255 * (np.log(1 + pixel_value / 255) / np.log(2))))

def power_transformation(pixel_value, gamma=2):
    return int(255 * (pixel_value / 255) ** gamma)

def piecewise_linear_transformation(pixel_value, threshold=127, slope=2):
    if pixel_value <= threshold:
        return int(slope * pixel_value)
    else:
        return pixel_value

specific_image = Image.new('RGB', (height, width))
logarithmic_image = Image.new('RGB', (height, width))
power_image = Image.new('RGB', (height, width))
linear_transform_image = Image.new('RGB', (height, width))

for y in range(height):
    for x in range(width):
        r, g, b = img_data[x, y]
        logarithmic_image.putpixel((x, y), (logarithmic_transformation(r), logarithmic_transformation(g), logarithmic_transformation(b)))
        power_image.putpixel((x, y), (power_transformation(r), power_transformation(g), power_transformation(b)))
        linear_transform_image.putpixel((x, y), (piecewise_linear_transformation(r), piecewise_linear_transformation(g), piecewise_linear_transformation(b)))
        specific_image.putpixel((x, y), (logarithmic_transformation(r) + power_transformation(r) + piecewise_linear_transformation(r), logarithmic_transformation(g) + power_transformation(g) + piecewise_linear_transformation(g), logarithmic_transformation(b) + power_transformation(b) + piecewise_linear_transformation(b)))

images = [(modified_img, 'Modifed image'), (specific_image, 'Specific image'), (power_image, 'Power transform image'), (linear_transform_image, 'Linear transform image'), (logarithmic_image, 'Logarithmic transform image')]

fig, axs = plt.subplots(1, 5, figsize=(14, 4))

for i, (img_p, img_t) in enumerate(images):
    axs[i].imshow(img_p)
    axs[i].set_title(img_t)
    axs[i].axis('off')

plt.subplots_adjust(hspace=1)
plt.show()
