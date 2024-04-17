from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"

#response = requests.get(url)
#response.raise_for_status()  # Raises stored HTTPError, if one occurred

# Open the image from the bytes of the response content
#image = Image.open(BytesIO(response.content))
image = Image.open('test.jpg')
#image.show()  # This will open the image using an image viewer
#plt.imshow(image)
#plt.axis('off')
#plt.show()

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
		f"Detected {model.config.id2label[label.item()]} with confidence "
		f"{round(score.item(), 3)} at location {box}"
    )
    top_left_corner = (box[0], box[1])  # x, y of the top left corner
    bottom_right_corner = (box[2], box[3])  # x, y of the bottom right corner
    draw.rectangle([top_left_corner, bottom_right_corner], outline="red", width=3)

image.save('out.jpg')
#image.show()