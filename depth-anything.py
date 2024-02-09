from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
# if Key error: "depth_anything" --> pip install git+https://github.com/huggingface/transformers.git

# load image
img_path = "/media/eric/D/cloud_eric/images/gas/IMG_20170716_185242.jpg"
image = Image.open(img_path)

# inference
depth = pipe(image)["depth"]

# show
plt.imshow(depth)
plt.show()
