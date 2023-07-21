
echo 
echo   " Running evaluation using 'validation_data'"

touch ./evaluation_loss
find "./output_grained_trained_models" -type f | sort -V | xargs python ./run_evaluation.py ./coarse_grained_graphs validation_data > ./evaluation_loss