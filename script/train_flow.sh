#echo "python3 train.py --id $1 --config $1_config.json --data_path ../data --model_path ../models"
#echo "python3 predict.py --id $1 --data_path ../data --model_path ../models --pred_path ../preds"
#echo "python3 submit.py --id $1 --data_path ../data --output_filepath ../submission_$1.csv"

#python3 train.py --id $1 --config $1_config.json --data_path ../data --model_path ../models && \
python3 predict.py --id $1 --data_path ../data --model_path ../models --pred_path ../preds && \
python3 submit.py --id $1 --data_path ../data --output_filepath ./$1_submission.csv --pred_path ../preds
