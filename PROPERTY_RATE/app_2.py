from flask import Flask, render_template,request

import pickle

import numpy as np

model=pickle.load(open('iri.pkl','rb'))

app=Flask(__name__,template_folder=r'C:\Users\Dell\3D Objects\PROPERTY_RATE')

@app.route('/')
def man():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
	data1=request.form['a']
	data2=request.form['b']
	data3=request.form['c']
	data4=request.form['d']
	data5=request.form['e']
	arr=np.array([[data1,data2,data3,data4,data5]])
	pred=model.predict(arr)
	print(len(pred))
	listToStr = ' '.join([str(elem) for elem in pred]) 
	print(listToStr) 
	strtoint=round(float(listToStr))
	print(strtoint)

	return render_template('after.php',data=strtoint)




if __name__=="__main__":
	app.run(debug=True)