pretraining_dataset_name=$1

python -m src.precompute.aggregate_cooccurrence_matrix --pretraining_dataset_name $pretraining_dataset_name
