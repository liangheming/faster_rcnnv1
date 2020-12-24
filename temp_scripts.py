import torch
from nets.faster_rcnn import FasterRCNN
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           max_thresh=768,
                           # debug=50
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=dataset.collect_fn)
    net = FasterRCNN()
    for img_input, valid_size,targets, batch_len in tqdm(dataloader):
        out = net(img_input, valid_size=valid_size, targets={"target": targets, "batch_len": batch_len})
        break
