config = {
	# Parameters to change during runs
	"dataset": "BCV", # COCO, BCV, CT_ORG, DECATHLON
	"split": 0, # 0-3 for coco
	"test_dataset": "",
	"lists_path": "../lists/",

	"mode": "train", # or test/val
	"shot": 3,
	"unlabelled": 6,
	"model_name": "Custom", # GCN, GAT, GCN2, GraphConv, GIN
	# "model_params": {

	# },
	"num_neighbors": 2,
	"alpha": 1,
	"beta": 1,

	"embedding_dim": 32,

	# CL variables
	"num_anchors": 1000,
	"num_samples": 1000,
	"temp": 0.07, # (default value used in a lot of places)
	"unsup_weight": 0.5,
	"batch_size": 4,
	"n_episodes": 100,

	# Medical Dataset paths
	"bcv_path": "",
	"ctorg_path": "",
	"decathlon_path": "",

	# General Dataset paths
	"coco_path": "",
	# "bcv_path": "",

	# optimizer params
	"init_lr": 0.001,
	"weight_decay": 1e-6, 

	"classes": {
		"COCO": 80,
		"BCV": 7,
		"CT-ORG": 6,
		"DECATHLON": 6, 
	}
	
}