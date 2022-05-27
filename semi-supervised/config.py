config = {

	# Parameters to change during runs
	"dataset": "COCO", # COCO, BCV, CT_ORG, DECATHLON
	"split": 0, # 0-3 for coco
	"test_dataset": "",
	"lists_path": "../lists",

	"mode": "train", # train or val
	"run_id_to_test": 51,
	"n_queries": 1,
	"ways": 1,
	"shot": 5,
	"unlabelled": 6,
	"model_name": "Custom", # GCN, GAT, GCN2, GraphConv, GIN
	# "model_params": {
	"batch_size": 1,
	"n_episodes": 4000,

	# },
	"num_neighbors": 1,
	"alpha": 1,
	"beta": 1,
	"edge_weight_function": 'mean-abs', # mean-abs or euclidean or deltaE
	"embedding_dim": 32,

	# CL variables
	"num_anchors": 1000,
	"num_samples": 2112,
	"temp": 0.1,
	"unsup_weight": 0.5,

	# Supervised CL variables 
	"num_positives": 64 ,
	"num_negatives": 2048, 

	# Unsupervised CL variables
	"unsup_anchors": 2048,

	# Medical Dataset paths
	"bcv_path": "",
	"ctorg_path": "",
	"decathlon_path": "",

	# General Dataset paths
	"coco_path": "../../MSCOCO/",
	# "bcv_path": "",

	# optimizer params
	"init_lr": 0.1,
	"weight_decay": 1e-6, 

	"classes": {
		"COCO": 80,
		"BCV": 7,
		"CT-ORG": 6,
		"DECATHLON": 6, 
	}
	
}
