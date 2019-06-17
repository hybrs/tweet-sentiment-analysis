import operator
import pickle
import sys
import numpy as np

grid_ = [ pickle.load(open('./cv_result/cv_result_'+str(i), 'rb')) for i in range(2) ]

grid_result = dict({})

for ds in grid_:
	for model_tag in ds.keys():
		fold_scores = ds[model_tag] 
		grid_result.update({
			model_tag: [np.average(fold_scores),np.std(fold_scores)]
			})


print("model_tag\tValidation accuracy +/-std")
print("========================================================")
for item in sorted(grid_result.items(), key = operator.itemgetter(1), reverse=True):
	print(item[0]+"\t"+str(item[1][0])[:5]+" +/-"+str(item[1][1])[:5])
	print("--------------------------------------------------------------------------")

#pickle.dump(file=open('grid_result_'+suffix, 'wb'), obj=grid_result)