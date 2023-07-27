cd "$(dirname $0)"
rm -rf  ./graphs_trajrep
echo 
echo " prepare directory and separate graphs"
tar -xf graphs_trajrep.tar.bz2


find ./graphs -type f -name '*_nodes.csv' \
| sort \
| while read -r NODESFILE
do
	GRAPHID="$(basename ${NODESFILE} _nodes.csv)"
	LINKSFILE="$(dirname ${NODESFILE})/${GRAPHID}_links.csv"
	OUTDIR="./graphs_trajrep/separated_graphs/${GRAPHID}"
	mkdir -p "$OUTDIR"
	cp "$NODESFILE" "$LINKSFILE" "$OUTDIR"
done

echo 
echo " separate data into train-vald-test \
      write std and mean values of training data "
mkdir -p ./graphs_trajrep/train_data
mkdir -p ./graphs_trajrep/validation_data
python ./utils.py ./graphs_trajrep


echo 
echo " prepare normalized input data "

python ./input_data.py ./graphs_trajrep
