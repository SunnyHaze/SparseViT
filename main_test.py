import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 4)
from datasets import mani_dataset
from SparseViT_Mul import SparseViT_Mul

input_dir = "./images/"
dataset = mani_dataset(
    path = input_dir,
    if_return_shape=True
)
print(dataset)
print(f":ength of this dataset: {len(dataset)}")
# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device:", device)

ckpt_path = "./checkpoint/checkpoint-160.pth"
model = SparseViT_Mul()

model.load_state_dict(
    torch.load(ckpt_path)['model'],
    strict = False
)
model = model.to(device)

results = []
model.eval()
with torch.no_grad():
    for img, gt, shape in dataset:   
        img, gt = img.to(device), gt.to(device)
        img = img.unsqueeze(0) # CHW -> 1CHW
        gt = gt.unsqueeze(0)
        # inference
        mask_pred = model(img, gt)
        output = mask_pred
        output = output[0].permute(1, 2, 0).cpu().numpy()
        if output.shape[-1] == 1:
            output = np.squeeze(output, axis=-1)
        pillow_img = Image.fromarray(output)
        output_resized = pillow_img.resize((shape[1], shape[0]), Image.LANCZOS)
        output_resized = np.array(output_resized)
        results.append(output_resized)      
print("Done!")
for i, (img, gt, res) in enumerate(zip(dataset.tp_path, dataset.gt_path, results)):
    img = plt.imread(img)
    gt = plt.imread(gt)
    plt.subplot(1, 3, 1)
    plt.title("Manipulated")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title("Groundtruth")
    plt.imshow(gt, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Predict Mask")
    plt.imshow(res, cmap="gray")
    plt.savefig('sample_'+ str(i) + '.png')
    plt.show()