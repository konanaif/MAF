pretraining_dataset_name=$1
filename=$2

python -m src.precompute.compute_occurrence_matrix --pretraining_dataset_name $pretraining_dataset_name --filename $filename
