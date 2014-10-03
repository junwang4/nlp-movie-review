#####
# terminal> sh run.sh
#####

#python experiment.py -h
#python experiment.py --cv 3 --pooling min
#python experiment.py --data_folder="/home/ivan" --play_ratio=0.01
python experiment.py --data_folder="../data" --play_ratio=1.0 --cv 10

