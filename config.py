config = {
	# Parameters to change during runs
	"dataset": "BCV",
	"split": 0, # 0-3 for coco
	"test_dataset": "",
	"lists_path": "../lists/"

	"mode": "train", # or test/val
	"shot": 3,
	"model_name": "Custom", # GCN, GAT, GCN2, GraphConv, GIN
	# "model_params": {

	# },
	"num_neighbors": 2,
	"alpha": 1,
	"beta": 1,


	# Medical Dataset paths
	"bcv_path": "",
	"ctorg_path": "",
	"decathlon_path": "",

	# General Dataset paths
	"coco_path": "",
	# "bcv_path": "",
}