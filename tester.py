import time
import numpy
import pickle

with open('vocablury.txt','r') as file_handler:
	vocablury=pickle.load(file_handler)

with open('y_labels.txt','r') as file_handler:
	y_labels=pickle.load(file_handler)

Phi_List=numpy.load('Phi_List.npy')

Theta_Matrix=numpy.load('Theta_Matrix.npy')

print("Enter the line to be tested")

inputLine = input()

list_of_all_lines = [inputLine]

Matrix_X=[]

for i in list_of_all_lines:
	tmp=i.lower().split()

	# Making a dictionary in which every element is mapped to 0
	dictionary_for_every_line=dict.fromkeys(vocablury,0)

	for j in tmp:
		if(j in vocablury):
			dictionary_for_every_line[j]=dictionary_for_every_line[j]+1
	# Extracting the list of values in the counter dictionary
	row_x=dictionary_for_every_line.values()
	Matrix_X.append(row_x)
	# Making a similar dictionary for y_labels in which the label which 
	# is tagged will be given a value 1

Matrix_X=numpy.array(Matrix_X)

H_Matrix = numpy.log(Phi_List)+(Matrix_X.dot(numpy.log(Theta_Matrix)))

for i in H_Matrix:
	max_element=numpy.amax(i)
	count=0
	for j in i:
		count=count+1
		if(j==max_element):
			print("The output class is "+ y_labels[count])
			break
