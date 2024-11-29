pretraining_dataset_name=$1

python -m src.precompute.aggregate_occurrence_matrix --pretraining_dataset_name $pretraining_dataset_name
