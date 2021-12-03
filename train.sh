for i in {11022..11022}
do
	bash tools/dist_train_4.sh configs/positional_encoding_in_gans/${i}_pe.py 1 --work-dir ./work_dirs/${i}
done 
