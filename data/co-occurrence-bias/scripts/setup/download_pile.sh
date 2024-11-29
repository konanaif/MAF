mkdir -p "data/pile"

for NUM in {00..29}
do
    nohup wget "https://the-eye.eu/public/AI/pile/train/"$NUM".jsonl.zst" -P "data/pile" > "logs/log.download_pile_train_"$NUM &
done

nohup wget "https://the-eye.eu/public/AI/pile/val.jsonl.zst" -P "data/pile" > "logs/log.download_pile_val" &
nohup wget "https://the-eye.eu/public/AI/pile/test.jsonl.zst" -P "data/pile" > "logs/log.download_pile_test" &
nohup wget "https://the-eye.eu/public/AI/pile/SHA256SUMS.txt" -P "data/pile" > "logs/log.download_pile_sha256" &
