from PIL import Image
from torchvision import transforms
import numpy as np

img_a = './n04599235/n04599235_140.JPEG'
img_b = './n07831146/n07831146_4.JPEG'

transform_forshow = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

img_a = np.asarray(transform_forshow(Image.open(img_a)))
img_b = np.asarray(transform_forshow(Image.open(img_b)))

img_mix = 0.9 * img_a + 0.1 * img_b
img_mix = Image.fromarray(img_mix.astype(np.uint8))
img_mix.save("mixup_0_9.jpg")
Image.fromarray(img_a.astype(np.uint8)).save("original_a.jpg")
Image.fromarray(img_b.astype(np.uint8)).save("original_b.jpg")