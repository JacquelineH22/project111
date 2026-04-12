import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import cv2
import numpy as np
from tqdm import tqdm

from pathlib import Path

import sys
sys.path.append('/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/gigatime/GigaTIME/scripts/')

import archs


def get_patches(image: np.ndarray, window_size: tuple = (512, 512), stride: tuple = (128, 128)) -> tuple:
    """Get sliding window patches.

    Args:
        image (np.ndarray): Image.
        window_size (tuple): Window size (default is (512, 512)).
        stride (tuple): Stride (default is (128, 128)).

    Returns:
        tuple: patches, positions (top left corners), and the original image size.

    """
    H, W = image.shape[:2]
    win_h, win_w = window_size
    stride_h, stride_w = stride

    pad_h = (stride_h - (H - win_h) % stride_h) % stride_h if H > win_h else win_h - H
    pad_w = (stride_w - (W - win_w) % stride_w) % stride_w if W > win_w else win_w - W
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    patches = []
    positions = []  # top left corner, (h, w)

    h = 0
    while h <= padded.shape[0] - win_h:
        w = 0
        while w <= padded.shape[1] - win_w:
            patch = padded[h: h + win_h, w: w + win_w]
            patches.append(patch)

            orig_h = min(h, H - 1)
            orig_w = min(w, W - 1)
            positions.append((orig_h, orig_w))

            w += stride_w
        h += stride_h

    return patches, positions, (H, W)


def do_inference(dataloader: DataLoader, model: torch.nn.Module, device: str = 'cuda:0') -> torch.Tensor:
    """Do inference.

    Args:
        dataloader (Dataloader): Dataloader.
        model (torch.nn.Module): GigaTIME model.
        device (str): Device to run.

    Returns:
        torch.Tensor: Outputs.

    """
    outputs = []
    model.eval()
    for patches in tqdm(dataloader):
        with torch.no_grad():
            outputs.append(model(patches.to(device)).cpu())
    outputs = torch.cat(outputs, dim=0)

    return outputs


def stitch_outputs(outputs: torch.Tensor, positions: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Stich sliding window outputs.

    Arg:
        outputs (torch.Tensor): Outputs, shape (B, C, h, w).
        positions (torch.Tensor): Top left corners of patches, shape (B, 2).
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        torch.Tensor: Stitched outputs, shape (C, H, W).

    """
    n_patches, n_channels, patch_h, patch_w = outputs.shape
    sum_logits = np.zeros((n_channels, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for i in tqdm(range(n_patches)):
        output = outputs[i].numpy()
        position = positions[i].numpy()

        h_start = position[0]
        w_start = position[1]
        h_end = min(h_start + patch_h, H)
        w_end = min(w_start + patch_w, W)
        actual_h = h_end - h_start
        actual_w = w_end - w_start

        sum_logits[:, h_start:h_end, w_start:w_end] += output[:, :actual_h, :actual_w]
        counts[h_start:h_end, w_start:w_end] += 1.0

    counts[counts == 0] = 1.0
    sum_logits = sum_logits / counts

    return torch.tensor(sum_logits)


class SlideDataset(Dataset):
    def __init__(self, image: np.ndarray, transforms: v2.Transform = None) -> None:
        """Dataset for a slide image.

        Args:
            image (np.ndarray): Image.
            transforms (v2.Transform): Transforms to be applied on a patch (default is None).

        Returns:
            None.

        """
        patches, positions, (H, W) = get_patches(image, window_size=(256, 256), stride=(128, 128))
        for i in range(len(patches)):
            patches[i] = torch.tensor(patches[i].transpose(2, 0, 1), dtype=torch.uint8)  # HWC -> CHW

        self.patches = patches
        self.positions = positions
        self.H = H
        self.W = W
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if transforms is None else transforms

        return

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.patches[idx]
        return self.transforms(patch)


if __name__ == '__main__':

    image_file = '/data1/hounaiqiao/wzr/Simulated_Xenium/luca/HE_aligned/wholeslide/align_he.png'
    checkpoint_file = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/gigatime/checkpoints/model.pth'
    save_dir = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/interval/results/luca/gigatime/'

    batch_size = 512
    num_workers = 16
    device = 'cuda:2'

    config = {
        'arch': 'gigatime',
        'input_channels': 3,
        'num_classes': 23,
    }

    # load data
    print('Loading data...')
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

    # create dataloader
    dataset = SlideDataset(image)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    # load model
    print('Loading model...')
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'])
    state_dict = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    # infer
    print('Inferring...')
    outputs = do_inference(dataloader, model, device=device)
    whole_outputs = stitch_outputs(outputs, torch.tensor(dataset.positions), dataset.H, dataset.W)
    whole_predictions = (torch.sigmoid(whole_outputs) > 0.8).float()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # torch.save(outputs.cpu(), '%s/patch_outputs.pth' % save_dir)
    # torch.save(whole_outputs.cpu(), '%s/whole_slide_outputs.pth' % save_dir)
    torch.save(whole_predictions.cpu(), '%s/whole_slide_predictions.pth' % save_dir)

    print('Done!')
