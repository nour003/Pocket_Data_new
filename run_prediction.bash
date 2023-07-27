cd "$(dirname $0)"

echo
echo "delete prev directory"
rm -rf ./output_predicted_values ./__pycache__ ./validation_data ./training_data
mkdir ./output_predicted_values

echo 
echo " prediction on training set and validation set"

for set_name in training_data validation_data
do
echo
echo " create direcory "
mkdir ./output_predicted_values/${set_name}
    for epoch in  5 10 20 60 80 100 120 180 300
    do
        mkdir ./output_predicted_values/${set_name}/epoch${epoch}.pth

        while read -r graph_name
        do
            graph_name="$(echo ${graph_name} | tr -d '\r' | tr -d '\r')"
            touch "./output_predicted_values/${set_name}/epoch${epoch}.pth/$graph_name"
        done < "./coarse_grained_graphs/${set_name}_files_prefixes"
        
        sorted_graphs="$(cat ./coarse_grained_graphs/${set_name}_files_prefixes | tr -d '\r'| sort)"
        python ./run_prediction.py ./coarse_grained_graphs ${set_name} "epoch${epoch}.pth" $sorted_graphs > ${set_name}_loss
    done
done
