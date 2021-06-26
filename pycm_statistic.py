import pycm


label = []	
result = []

cm = pycm.ConfusionMatrix(actual_vector=label, predict_vector=result)

print(cm)