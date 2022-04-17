from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import json
from config import config

# config = json.load("../config.json")

parts = ['train2017', 'val2017'] # use both train and val

for split in parts:
	annFile = config['coco_path'] + 'annotations/instances_{}.json'.format(split)
	coco = COCO(annFile) # init coco annotation file
	indxs = coco.getImgIds() # get image ids
	coco_ids = coco.getCatIds() # get category ids for the images
	continual_ids = np.arange(1, 81) # because there are 1-80 classes

	id_mapping = {coco_id:con_id for coco_id, con_id in zip(coco_ids, continual_ids)}
	# save to not preproc multiple times
	save_dir = config['coco_path'] +'annotations/coco_masks/instance_{}'.format(split)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for indx in tqdm(indxs):
		img_meta = coco.loadImgs(indx)[0]
		annIds = coco.getAnnIds(imgIds=img_meta['id'])
		anns = coco.loadAnns(annIds)
		semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
		for ann in anns:
			if ann['iscrowd']:
				continue
			catId = ann['category_id']
			mask = coco.annToMask(ann)
			semantic_mask[mask == 1] = id_mapping[catId]
		mask_img = Image.fromarray(semantic_mask)
		mask_img.save(os.path.join(save_dir, img_meta['file_name'].replace('jpg', 'png')))