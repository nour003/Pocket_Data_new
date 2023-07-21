cd "$(dirname $0)"

echo
echo "delete prev directory"
rm -rf ./output_predicted_values ./__pycache__
mkdir ./output_predicted_values

echo 
echo " prediction on training set"
echo
echo " crate direcories "
mkdir ./output_predicted_values/train_set

for epoch in 20 40 80 160 320
do
    mkdir ./output_predicted_values/train_set/epoch${epoch}.pth

    echo
    echo " crate direcories "

    while read -r graph_name
    do
        graph_name="$(echo ${graph_name} | tr -d '\r' | tr -d '\r')"
        touch "./output_predicted_values/train_set/epoch${epoch}.pth/$graph_name"
    done < "./coarse_grained_graphs/training_data_files_prefixes"
    
    sorted_graphs="$(cat ./coarse_grained_graphs/training_data_files_prefixes | tr -d '\r'| sort)"
    python ./run_prediction.py ./coarse_grained_graphs training_data "epoch${epoch}.pth" $sorted_graphs > grained_evaluation_loss
done

echo 
echo " Done"
