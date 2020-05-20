cd ..

mkdir -p train_data/trash/
mkdir -p val_data/trash/
mkdir -p test_data/trash/
mkdir -p train_data/not_trash/
mkdir -p val_data/not_trash/
mkdir -p test_data/not_trash/

num_files=$(ls labeled_data/trash | wc -l)
num_train=$((3*num_files/5))
num_val=$((1*num_files/5))
num_test=$((num_files-num_train-num_val))
echo "$num_files"
echo "$num_train"
echo "$num_val"
echo "$num_test"

ls labeled_data/trash/ | sort -t "+" -n -k2 | sed -n "1,$num_train p" | xargs -I '{}' cp labeled_data/trash/{} train_data/trash/
ls labeled_data/trash/ | sort -t "+" -n -k2 | sed -n "$((num_train+1)),$((num_train+num_val)) p" | xargs -I '{}' cp labeled_data/trash/{} val_data/trash/
ls labeled_data/trash/ | sort -t "+" -n -k2 | sed -n "$((num_train+num_val+1)),$num_files p" | xargs -I '{}' cp labeled_data/trash/{} test_data/trash/

ls labeled_data/not_trash/ | sort -t "+" -n -k2 | sed -n "1,$num_train p" | xargs -I '{}' cp labeled_data/not_trash/{} train_data/not_trash/
ls labeled_data/not_trash/ | sort -t "+" -n -k2 | sed -n "$((num_train+1)),$((num_train+num_val)) p" | xargs -I '{}' cp labeled_data/not_trash/{} val_data/not_trash/
ls labeled_data/not_trash/ | sort -t "+" -n -k2 | sed -n "$((num_train+num_val+1)),$num_files p" | xargs -I '{}' cp labeled_data/not_trash/{} test_data/not_trash/
