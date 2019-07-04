import sys
import numpy as np

def main():
	# read training file
#	f_train_path = 'toydata/toytrain.tsv'
#	f_validation_path = 'toydata/toyvalidation.tsv'
#	f_test_path = 'toydata/toytest.tsv'

#	f_train_path = 'largedata/train.tsv'
#	f_validation_path = 'largedata/validation.tsv'
#	f_test_path = 'largedata/test.tsv'

#	Model = '2'
#	epoch = 2
	f_train_path = sys.argv[1]
	f_validation_path = sys.argv[2]
	f_test_path = sys.argv[3]
	f_trainlabels_path = sys.argv[4]
	f_testlabels_path = sys.argv[5]
	f_modelmetrics_path = sys.argv[6]
	epoch = int(sys.argv[7])
	Model = sys.argv[8]

	train = file_data(f_train_path)
	validation = file_data(f_validation_path)
	test = file_data(f_test_path)

	attributes = np.unique(train.x)
	labels = np.sort(np.unique(train.y)) # sort the list to ensure that for ties the one with lower ASCII will be the output

	# convert train, validation and test data into vectors
	v_train = vectorize(train, attributes, labels, Model)
	v_validation = vectorize(validation, attributes, labels, Model)
	v_test = vectorize(test, attributes, labels, Model)

	if Model == '1':
		M = len(attributes) + 1 # with bias term 
	elif Model == '2':
		M = len(attributes) * 3 + 1

	modelmetrics = ''

	K = len(labels)
	theta = np.zeros((K, M))
	eta = 0.5

	# train and generate metrics
	for i in range(epoch):

		theta = SGD(v_train.x, v_train.y, theta, eta)

		J_train = J(v_train.x, v_train.y, theta)
		J_validation = J(v_validation.x, v_validation.y, theta)

		modelmetrics += 'epoch={} likelihood(train): {:.6f}\n'.format(i+1, J_train)
		modelmetrics += 'epoch={} likelihood(validation): {:.6f}\n'.format(i+1, J_validation)

	p_train = predict(v_train.x, theta)
	error_train = 1 - float(sum(p_train==v_train.y))/len(v_train.x)
	modelmetrics += 'error(train): {:.6f}\n'.format(error_train)

	p_test = predict(v_test.x, theta)
	error_test = 1 - float(sum(p_test==v_test.y))/len(v_test.x)
	modelmetrics += 'error(test): {:.6f}\n'.format(error_test)

	# write labels and labels
	labels_train = ''
	counter = 0
	for line in train.lines:
		if line != '\n':
			labels_train += labels[p_train[counter]]
			labels_train += '\n'
			counter += 1
		else:
			labels_train += '\n'

	labels_test = ''
	counter = 0
	for line in test.lines:
		if line != '\n':
			labels_test += labels[p_test[counter]]
			labels_test += '\n'
			counter += 1
		else:
			labels_test += '\n'

	with open(f_modelmetrics_path, 'w') as f:
		f.write(modelmetrics)
	f.closed
	with open(f_trainlabels_path, 'w') as f:
		f.write(labels_train)
	f.closed
	with open(f_testlabels_path, 'w') as f:
		f.write(labels_test)
	f.closed
'''
# for-loop version
def J(x, y, theta):
	N = x.shape[0]
	M = x.shape[1]
	K = theta.shape[0]

	result = 0
	for i in range(N):
		for k in range(K):
			result += int(y[i]==k) * np.log( np.exp(np.dot(x[i], theta[k])) / sum(np.exp(np.dot(x[i], theta.T))) )
#			if y[i]==k:
#				print(attributes[np.argmax(x[i])], labels[k])
	result = -(1./N) * result

	return result

def DJ(x, y, theta, i, k): # pass the data number i as well for DJ(i)(theta)
	N = x.shape[0]
	M = x.shape[1]
	K = theta.shape[0]
	result = -( int(y[i]==k) - np.exp(np.dot(theta[k], x[i])) / sum(np.exp(np.dot(x[i], theta.T))) ) * x[i]

	return result

def SGD(x, y, theta, eta): # one step of theta update
	N = x.shape[0]
	M = x.shape[1]
	K = theta.shape[0]

	for i in range(N):
		g = np.zeros(theta.shape)
		for k in range(K):
			g[k] = DJ(x, y, theta, i, k)
		theta = theta - eta*g

	return theta

'''
# vectorized version
def J(x, y, theta):
	N = x.shape[0]
	M = x.shape[1]
	K = theta.shape[0]

	result = 0
	II = np.zeros((N, K))
	for i in range(N):
		II[i][y[i]] = 1
	result = -(1./N) * ( II.T * (np.dot(x, theta.T).T - np.log(np.exp(np.dot(x, theta.T)).sum(axis=1))) ).sum()

	return result

def SGD(x, y, theta, eta):
	N = x.shape[0]
	M = x.shape[1]
	K = theta.shape[0]
	for i in range(N):
		II = np.zeros(K)
		II[y[i]] = 1
		DJ_i = - np.outer(( II - np.exp(np.dot(x[i], theta.T)) / np.exp(np.dot(x[i], theta.T)).sum() ), x[i])
		theta = theta - eta * DJ_i
	return theta

def predict(x, theta):

	max = np.argmax(np.dot(x, theta.T), axis=1)
	return max

class file_data:

	def __init__(self, file_path):
		self.x = []
		self.y = []
		self.lines = []

		data = []
		data.append(['BOS','O'])
		with open(file_path, 'r') as f_open:
			for line in f_open:
				self.lines.append(line)
				if line != '\n':
					data.append(line.replace('\n', '').split('\t'))
				else:
					data.append(['EOS','O'])
					data.append(['BOS','O'])
		data.append(['EOS','O'])

		for l in data:
			self.x.append(l[0])
			self.y.append(l[1])

class vectorize:

	def __init__(self, data, attributes, labels, Model):

		K = len(labels)

		if Model == '1':

			sparse_x = []
			sparse_y = []
			for i, row in enumerate(data.x):
				if (row != 'EOS') and (row != 'BOS'):
					for j, input in enumerate(attributes):
						if input == data.x[i]:
							sparse_x.append(j)
							continue
					for j, label in enumerate(labels):
						if label == data.y[i]:
							sparse_y.append(j)
							continue

			N = len(sparse_x)			
			M = len(attributes) + 1 # with bias term 
			self.x = np.zeros((N, M), dtype=int)
			for i, row in enumerate(sparse_x):
				self.x[i, row] = 1
				self.x[i, -1] = 1
			self.y = np.array(sparse_y)

		elif Model == '2':

			sparse_x = []
			sparse_y = []
			for i, row in enumerate(data.x):
				if (row != 'EOS') and (row != 'BOS'):
					for j, input in enumerate(attributes):
						if input == data.x[i-1]:
							x1 = j
						elif input == data.x[i]:
							x2 = j
						elif input == data.x[i+1]:
							x3 = j
					sparse_x.append([x1, x2, x3])

					for j, label in enumerate(labels):
						if label == data.y[i]:
							sparse_y.append(j)
							continue

			N = len(sparse_x)
			M = len(attributes) * 3 + 1 # three row vectors for prrev, curr, next and a one bias term 
			self.x = np.zeros((N, M), dtype=int)
			for i, row in enumerate(sparse_x):
				for j, col in enumerate(row):
					self.x[i, j * len(attributes) + col] = 1 # prev, curr, next
				self.x[i, -1] = 1
			self.y = np.array(sparse_y)

if __name__ == "__main__":
    main()
