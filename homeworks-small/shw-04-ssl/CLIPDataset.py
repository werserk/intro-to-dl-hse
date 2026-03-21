import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import typing as tp

class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        :tokenizer -- LM Tokenizer 
        """
        self.max_tokenizer_length = 200
        self.truncation = True
        self.padding = "max_length"
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer
        
        self.encoded_captions = self.tokenizer(
            self.captions,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_tokenizer_length,
            return_tensors=None
        )
        
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:
        """
        This one should return dict(keys=['image', 'caption', 'input_ids', 'attention_mask'], value=[Image, Caption, ...])
        """
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }
        
        image_filename = self.image_filenames[idx]
        image = Image.open(f"{self.image_path}/{image_filename}").convert("RGB")
        item['image'] = self.transforms(image)
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)
