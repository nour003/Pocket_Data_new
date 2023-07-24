cd "$(dirname $0)"

echo
echo " plot the validation and train loss "
python ./plot_loss.py ./coarse_grained_graphs validation_data train_loss evaluation_loss
