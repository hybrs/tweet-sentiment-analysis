FILE="./configs"
while read CONFIG; do
	python cross_validation.py $CONFIG
done < "$FILE"