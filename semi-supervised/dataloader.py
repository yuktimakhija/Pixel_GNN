from torch.utils.data import Dataset
import glob
import numpy as np
import json
import cv2
import os
import torch
from tqdm import tqdm
import random
from img2graph import img2graph, support_graph_matrix
from config import config
from torchvision import transforms

# config = json.load("../config.json")

class GeneralDataLoader(Dataset):
	def __init__(self, sup_data_list, unsup_data_list):
		# sup_data_list is a list of 2-element lists of supervsied training data (20%)
		# it is [ [img,label], [img,label], ....] 
		# unsup_data_list is the same for unsupervsied training data (80%)
		# it is [ [img,label], [img,label], ....] (only using labels to differentiate training and testing)
		self.dataset = config['dataset']
		self.split = config['split']
		self.shot = config['shot']
		self.unsup = config['unlabelled']
		self.mode = config['mode']
		self.rng = np.random.default_rng()
		if self.dataset == 'COCO':
			self.path = config['coco_path']
			if config['split'] == 0:
				self.training_classes = list(range(21, 81)) # classes 21 to 80
				self.testing_classes = list(range(1, 21)) # classes 1 to 20
			elif self.split == 1:
				self.training_classes = list(range(1, 21)) + list(range(41, 81)) # classes 1 to 20 and 41 to 80
				self.testing_classes = list(range(21, 41)) # classes 21 to 40
			elif self.split == 2:
				self.training_classes = list(range(1, 41)) + list(range(61, 81)) # classes 1 to 40 and 61 to 80
				self.testing_classes = list(range(41, 61)) # classes 41 to 60
			elif self.split == 3: 
				self.training_classes = list(range(1, 61)) # classes 1 to 60
				self.testing_classes = list(range(61, 81)) # classes 61 to 80
			
		# elif dataset == ''
		else:
			raise ValueError("Wrong Dataset (General)")

		if self.mode == 'train':
			self.sup_data_list, self.sub_class_file_list, self.unsup_data_list =\
				make_dataset(self.split, self.path, sup_data_list, unsup_data_list, self.training_classes)
			assert len(self.sub_class_file_list.keys()) == len(self.training_classes)
		elif self.mode == 'val':
			self.sup_data_list, self.sub_class_file_list, self.unsup_data_list =\
				make_dataset(self.split, self.path, sup_data_list, unsup_data_list, self.testing_classes)
			assert len(self.sub_class_file_list.keys()) == len(self.testing_classes) 

	def __len__(self):
		return len(self.sup_data_list)

	def __getitem__(self, idx):
		# adapted from cyctr
		q = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])
		p = lambda x: p(x).permute(1,2,0)

		imgpath, lblpath = self.sup_data_list[idx]
		img = cv2.cvtColor(cv2.imread(imgpath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) 
		img = np.float32(img)
		label = cv2.imread(lblpath, cv2.IMREAD_GRAYSCALE)

		if img.shape[0] != label.shape[0] or img.shape[1] != label.shape[1]:
			raise (RuntimeError("Query Image & label shape mismatch: " + imgpath + " " + lblpath + "\n"))          
		label_classes = np.unique(label).tolist()
		if 0 in label_classes:
			label_classes.remove(0)
		if 255 in label_classes:
			label_classes.remove(255) 
		new_label_classes = []       
		for c in label_classes:
			if c in self.testing_classes:
				if self.mode == 'val' or self.mode == 'test':
					new_label_classes.append(c)
			if c in self.training_classes:
				if self.mode == 'train':
					new_label_classes.append(c)
		label_classes = new_label_classes    
		assert len(label_classes) > 0

		# Choose a random class for query
		class_chosen = label_classes[random.randint(1,len(label_classes))-1]

		# to mask the labels with chosen class
		target_pix = np.where(label == class_chosen)
		ignore_pix = np.where(label == 255)
		# reinitialize labels
		label[:,:] = 0
		if target_pix[0].shape[0] > 0:
			label[target_pix[0],target_pix[1]] = 1 
		label[ignore_pix[0],ignore_pix[1]] = 255           
		# Now label has been "masked" with the class_chosen


		file_class_chosen = self.sub_class_file_list[class_chosen]
		num_file = len(file_class_chosen)
		num_unsup_total = len(self.unsup_data_list)

		support_image_path_list = []
		support_label_path_list = []
		# support_idx_list = []
		
		# Choose support images randomly of the same class
		support_img_lbl_pairs = self.rng.choice(file_class_chosen, self.shot, replace=False)
		# check that query is not chosen for support
		while [imgpath,lblpath] in support_img_lbl_pairs:
			support_img_lbl_pairs = self.rng.choice(file_class_chosen, self.shot, replace=False)
	
		support_image_path_list = support_img_lbl_pairs[:,0]
		support_label_path_list = support_img_lbl_pairs[:,1]

		# Choose unsupervised support images randomly of any training class
		unsup_image_path_list = self.rng.choice(self.unsup_data_list, self.unsup, replace=False)
		
		# old code (cyctr)
		# for k in range(self.shot):
		# 	support_idx = random.randint(1,num_file)-1
		# 	support_image_path = imgpath
		# 	support_label_path = lblpath
		# 	# choose 1 random image
		# 	while((support_image_path == imgpath and support_label_path == lblpath) or support_idx in support_idx_list):
		# 		support_idx = random.randint(1,num_file)-1
		# 		support_image_path, support_label_path = file_class_chosen[support_idx]                
		# 	# add to support
		# 	support_idx_list.append(support_idx)
		# 	support_image_path_list.append(support_image_path)
		# 	support_label_path_list.append(support_label_path)
		
		# Now that we have paths for support images, read the images.
		support_image_list = []
		support_label_list = []
		subcls_list = []

		unsup_image_list = []

		for k in range(self.shot):
			if self.mode == 'train':
				subcls_list.append(self.training_classes.index(class_chosen))
			else:
				subcls_list.append(self.testing_classes.index(class_chosen))

			support_image_path = support_image_path_list[k]
			support_label_path = support_label_path_list[k] 
			# same as above
			support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
			support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
			support_image = np.float32(support_image)
			support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)

			# masking the labels for one support image, same as above
			target_pix = np.where(support_label == class_chosen)
			ignore_pix = np.where(support_label == 255)
			support_label[:,:] = 0
			support_label[target_pix[0],target_pix[1]] = 1 
			support_label[ignore_pix[0],ignore_pix[1]] = 255

			if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
				raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))     

			support_image_list.append(p(support_image))
			support_label_list.append(p(support_label))
		
		for k in range(self.unsup):
			# same as above
			unsup_image_path = unsup_image_path_list[k] 
			unsup_image = cv2.imread(unsup_image_path, cv2.IMREAD_COLOR)      
			unsup_image = cv2.cvtColor(unsup_image, cv2.COLOR_BGR2RGB)
			unsup_image = np.float32(unsup_image)
			
			unsup_image_list.append(p(unsup_image))
			
		# Now we should have all support images 
		assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    
		assert len(unsup_image_list) == self.unsup
		
		return support_image_list, support_label_list, unsup_image_list, [p(img)]
		# q_graph = img2graph(img, label)
		q_index, sup_index, sup_graph, unsup_graph, task_graph =\
			 support_graph_matrix(support_image_list, support_label_list, unsup_image_list, [img])
		if self.mode == 'train':
			return q_index, sup_index, sup_graph, unsup_graph, task_graph, torch.tensor(label.reshape(-1))
		else:
			return q_index, sup_index, sup_graph, unsup_graph, task_graph, torch.tensor(label.reshape(-1))


def make_dataset(split, path, data_list, unsup_data_list, training_classes):
	assert split in [0, 1, 2, 3, 10, 11, 999]
	if not os.path.isfile(data_list):
		raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

	split_data_list = data_list.split('.')[0] + '_split{}'.format(split) + '.pth'
	if os.path.isfile(split_data_list):
		image_label_list, sub_class_file_list, unsup_image_list = torch.load(split_data_list)
		return image_label_list, sub_class_file_list, unsup_image_list

	image_label_list = []
	unsup_image_list = []
	# list_read = open(data_list).readlines()
	list_read = json.load(open(data_list))
	unsup_list_read = json.load(open(unsup_data_list))
	print("Processing data...")
	sub_class_file_list = {}
	for sub_c in training_classes:
		sub_class_file_list[sub_c] = []

	print('Making dataset (supervised)')
	for l_idx in tqdm(range(len(list_read))):
		line = list_read[l_idx]
		line_split = line
		image_name = os.path.join(path, line_split[0])
		label_name = os.path.join(path, line_split[1])
		item = (image_name, label_name)
		label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
		label_class = np.unique(label).tolist()

		if 0 in label_class:
			label_class.remove(0)
		if 255 in label_class:
			label_class.remove(255)

		new_label_class = []       
		for c in label_class:
			if c in training_classes:
				tmp_label = np.zeros_like(label)
				target_pix = np.where(label == c)
				tmp_label[target_pix[0],target_pix[1]] = 1 
				if tmp_label.sum() >= 2 * 32 * 32:      
					new_label_class.append(c)

		label_class = new_label_class    

		if len(label_class) > 0:
			image_label_list.append(item)
			for c in label_class:
				if c in training_classes:
					sub_class_file_list[c].append(item)
	
	print('Making dataset (unsupervised)')
	for l_idx in tqdm(range(len(unsup_list_read))):
		line = unsup_list_read[l_idx]
		line_split = line
		image_name = os.path.join(path, line_split[0])
		label_name = os.path.join(path, line_split[1])
		label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
		label_class = np.unique(label).tolist()

		if 0 in label_class:
			label_class.remove(0)
		if 255 in label_class:
			label_class.remove(255)

		new_label_class = []       
		for c in label_class:
			if c in training_classes:
				tmp_label = np.zeros_like(label)
				target_pix = np.where(label == c)
				tmp_label[target_pix[0],target_pix[1]] = 1 
				if tmp_label.sum() >= 2 * 32 * 32:      
					new_label_class.append(c)

		label_class = new_label_class    

		if len(label_class) > 0:
			unsup_image_list.append(image_name)
					
	print("Checking image-label pair {} list done! ".format(split))

	print("Saving processed data...")
	torch.save((image_label_list, sub_class_file_list, unsup_image_list), split_data_list)
	print("Done")
	return image_label_list, sub_class_file_list, unsup_image_list

class MedicalDataLoader(Dataset):
	def __init__(self, img_paths, label_paths):
		self.dataset = config['dataset']
		if self.dataset == 'BCV':
			self.path = config['bcv_path']
		elif self.dataset == 'CT-ORG':
			self.path = config['ctorg_path']
		elif self.dataset == 'Decathlon':
			self.path = config['decathlon_path']
		else:
			raise ValueError("Wrong Dataset (Medical)")

		self.img_paths = img_paths
		self.label_paths = label_paths
		self.shot = config["shot"]
		self.mode = config['mode']
		self.length = config['n_iter']
		self.valid_img_n = len(img_paths)
		self.size = config['size']
		self.s_idx = config["s_idx"]
		self.img_lists = []
		self.slice_cnts = []
		# print(self.img_paths)
		for img_path in self.img_paths:
			fnames = os.listdir(img_path)
			# print(img_path)
			# print(len(fnames))
			self.slice_cnts.append(len(fnames))
			fnames = [int(e.split(".")[0]) for e in fnames]
			fnames.sort()
			fnames = [f"{e}.npy" for e in fnames]
			self.img_lists.append(fnames)

		if not self.is_train: # for testing
			self.length = sum(self.slice_cnts)

	def path_make(self, idx, option='train'):
		path_volume = f"{self.path}/{idx}/{option}"
		return glob(f"{path_volume}/img/*"), glob(f"{path_volume}/label/*")

	def paths(self, idx):
		tr_imgs, tr_labels = self.path_make(idx, 'train')
		val_imgs, val_labels = self.path_make(idx, 'valid')
		ts_imgs, ts_labels = self.path_make(idx, 'test')
		return tr_imgs, tr_labels, val_imgs, val_labels, ts_imgs, ts_labels
	
	def get_sample(self, s_img_paths_all, s_label_paths_all, q_img_paths, q_label_paths):
		s_imgs_all, s_labels_all = [],[]
		for s_idx, s_img_paths in enumerate(s_img_paths_all):
			s_label_paths = s_label_paths_all[s_idx]
			imgs, labels = [],[]

			for i in range(len(s_img_paths)):
				img_path, label_path = s_img_paths[i], s_label_paths[i]
				img = np.load(img_path)
				imgs.append(img)
				label = np.load(label_path)
				labels.append(label)

			s_imgs = np.stack(imgs,axis=0)
			s_labels = np.stack(labels,axis=0)
			s_imgs_all.append(s_imgs)
			s_labels_all.append(s_labels)

		s_imgs = np.stack(s_imgs_all,axis=0)
		s_labels = np.stack(s_labels_all,axis=0)

		imgs, labels = [],[]
		# Load query
		img_path, label_path
		img = np.load(img_path)
		imgs.append(img)
		label = np.load(label_path)
		labels.append(label)

		q_imgs = np.stack(imgs,axis=0)
		q_labels = np.stack(labels,axis=0)

		sample = {
			"s_x":torch.tensor(s_imgs),
			"s_y":torch.tensor(s_labels), #.long()
			"q_x":torch.tensor(q_imgs),
			"q_y":torch.tensor(q_labels), #.long()
			"s_fname":s_img_paths_all,
			"q_fname":q_img_paths,
		}
		return sample

	def handle_idx(self, s_n, q_idx, q_n):
		"""
		choose slices for support and query volume
		:return: supp_idx, qry_idx
		"""
		q_ratio = (q_idx)/(q_n-1)
		s_idx = round((s_n-1)*q_ratio)
		return s_idx

	def getitem_train(self):
		idx_space = [i for i in range(self.valid_img_n)]
		
		subj_idxs = random.sample(idx_space, self.shot+1)
		s_subj_idxs = subj_idxs[:self.shot]
		q_subj_idx = subj_idxs[self.shot]
		q_subj_img_path = self.img_paths[q_subj_idx]
		q_subj_label_path = self.label_paths[q_subj_idx]
		q_fnames = self.img_lists[q_subj_idx]
		q_idx = random.randrange(0, len(q_fnames))

		s_img_paths_all, s_label_paths_all = [],[]
		for s_subj_idx in s_subj_idxs:
			s_subj_img_path = self.img_paths[s_subj_idx]
			s_subj_label_path = self.label_paths[s_subj_idx]
			s_fnames = self.img_lists[s_subj_idx]

			## choose support and query slice
			s_idx = self.handle_idx(len(s_fnames), q_idx, len(q_fnames))
			s_fnames_selected = s_fnames[s_idx:s_idx+1]

			## define path, load data, and return
			s_img_paths_selected = [f"{s_subj_img_path}/{fname}" for fname in s_fnames_selected]
			s_label_paths_selected = [f"{s_subj_label_path}/{fname}" for fname in s_fnames_selected]
			s_img_paths_all.append(s_img_paths_selected)
			s_label_paths_all.append(s_label_paths_selected)

		q_fnames_selected = q_fnames[q_idx:q_idx + 1]
		q_img_paths_selected = [f"{q_subj_img_path}/{fname}" for fname in q_fnames_selected]
		q_label_paths_selected = [f"{q_subj_label_path}/{fname}" for fname in q_fnames_selected]
		to_return = self.get_sample(s_img_paths_all, s_label_paths_all, q_img_paths_selected, q_label_paths_selected)
		# for k in to_return.keys():
		#     to_print = to_return[k].shape if torch.is_tensor(to_return[k]) else to_return[k]
		#     print(k, ": ", to_print)
		return to_return

	def getitme_test(self, idx):
		q_subj_idx, q_idx = self.get_test_subj_idx(idx)
		q_subj_img_path = self.img_paths[q_subj_idx]
		q_subj_label_path = self.label_paths[q_subj_idx]
		q_fnames = self.img_lists[q_subj_idx]

		s_img_paths_all, s_label_paths_all = [],[]
		for s_idx in range(self.shot):
			s_subj_img_path = self.s_img_paths[s_idx]
			s_subj_label_path = self.s_label_paths[s_idx]
			s_fnames = self.s_fnames_list[s_idx]
			## choose support and query slice
			s_idx = self.handle_idx(len(s_fnames), q_idx, len(q_fnames))
			s_fnames_selected = s_fnames[s_idx:s_idx+1]
			## define path, load data, and return
			s_img_paths_selected = [f"{s_subj_img_path}/{fname}" for fname in s_fnames_selected]
			s_label_paths_selected = [f"{s_subj_label_path}/{fname}" for fname in s_fnames_selected]
			s_img_paths_all.append(s_img_paths_selected)
			s_label_paths_all.append(s_label_paths_selected)

		q_fnames_selected = q_fnames[q_idx:q_idx + 1]
		q_img_paths_selected = [f"{q_subj_img_path}/{fname}" for fname in q_fnames_selected]
		q_label_paths_selected = [f"{q_subj_label_path}/{fname}" for fname in q_fnames_selected]
		return self.get_sample(s_img_paths_all, s_label_paths_all, q_img_paths_selected, q_label_paths_selected)

	def get_len_train(self):
		return self.length

	def get_len_test(self):
		return self.length

	def get_val_subj_idx(self, idx):
		for subj_idx,cnt in enumerate(self.q_cnts):
			if idx < cnt:
				return subj_idx, idx*self.q_max_slice
			else:
				idx -= cnt

		print("get_val_subj_idx function is not working.")
		assert False

	def get_test_subj_idx(self, idx):
		for subj_idx,cnt in enumerate(self.slice_cnts):
			if idx < cnt:
				return subj_idx, idx
			else:
				idx -= cnt

		print("get_test_subj_idx function is not working.")
		assert False

	def get_cnts(self):
		## only for test loader
		return self.slice_cnts

	def img_load(self, img_path, seed=0):
		img_arr = np.load(img_path)
		return img_arr