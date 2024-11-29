for NUM in {00..29}
do
    nohup unzstd "data/pile/train/"$NUM".jsonl.zst" > "logs/log.extract_pile_train_"$NUM &
done

nohup unzstd "data/pile/val.jsonl.zst" > "logs/log.extract_pile_val" &
nohup unzstd "data/pile/test.jsonl.zst" > "logs/log.extract_pile_test" &
