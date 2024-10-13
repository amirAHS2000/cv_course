import timm
import torch

# load a pre-trained MobileNet model
model_name = "mobilenetv3_large_100"

model = timm.create_model(model_name, pretrained=True)

# use model for inference
model.eval()

# forward pass with a dummy input
# batch size 1, 3 color channels, 224x224 image
input_tensor = torch.rand(1, 3, 224, 224)

output = model(input_tensor)
print(output)