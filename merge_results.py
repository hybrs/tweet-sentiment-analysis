import operator
import pickle
import sys
import numpy as np

def print_sorted(scores, score):
	print("model_tag\t"+score+" +/-std")
	print("========================================================")
	for item in sorted(scores[score].items(), key = operator.itemgetter(1), reverse=True):
		print(item[0]+"\t"+str(item[1][0])[:5]+" +/-"+str(item[1][1])[:5])
		print("--------------------------------------------------------------------------")

		
def print_sortedTabMrk(scores, score):
    print("| <b>model </b>|  <b>"+score+"</b>  |")
    print("|:--------:|:--------------:|")
    for item in sorted(scores[score].items(), key = operator.itemgetter(1), reverse=True):
        model = item[0]
        avg = str(item[1][0])[:5]
        dev = str(item[1][1])[:5]
        print("| "+model+" | "+avg+" ±"+dev+" |")
    print("\n\n")


def print_sortedTabLtx(scores, score):
    print("\\begin{table}[]")
    print("\\begin{tabular}{|c|c|}")
    print("\\hline")
    print(" \\textbf{model}  & \\textbf{"+score+"} \\\\ \\hline")
    for item in sorted(scores[score].items(), key = operator.itemgetter(1), reverse=True):
        model = item[0]
        avg = str(item[1][0])[:5]
        dev = str(item[1][1])[:5]
        print("\\textit{"+model+"} & "+avg+" $\pm$"+dev+"\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n\n")
	
		
grid_ = [ pickle.load(open('./cv_result/cv_result_b32-e2-n('+str(i)+', '+str(i)+', '+str(i)+')-k(2, 3, 4)-xTW200', 'rb')) for i in [100, 150, 200, 250, 300, 350] ]

scores_ =  dict({
	'accuracy' : dict({}),
	'f1' : dict({}),
	'mavg_recall' :  dict({})
	})

for ds in grid_:
	for model_tag in ds.keys():
		scores = ds[model_tag]
		scores_['accuracy'].update({
			model_tag: [np.average(scores['accuracy']),np.std(scores['accuracy'])]
			})
		scores_['f1'].update({
			model_tag: [np.average(scores['f1']),np.std(scores['f1'])]
			})
		scores_['mavg_recall'].update({
			model_tag: [np.average(scores['mavg_recall']),np.std(scores['mavg_recall'])]
			})

for score_ in scores_.keys():
	print_sorted(scores_, score_)

if len(sys.argv) > 1 : pickle.dump(file=open('grid_result_'+sys.argv[1], 'wb'), obj=scores_)
