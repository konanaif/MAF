pretraining_dataset_name=$1 # pile
filename=$2 # please custom

python -m src.precompute.compute_term_document_index --pretraining_dataset_name $pretraining_dataset_name --filename $filename
