import pandas as pd
from visualize import visualize_3d
import statistics
import sys

class linearRegressor:
	def __init__(self):
		self.alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
		self.age = []
		self.weight = []
		self.height = []
		self.n = 0
		self.num_iter = 100
		self.weights = []
		self.fx = []

	def read(self, data):
		self.age = list(data['a'])
		self.weight = list(data['w'])
		self.height = list(data['h'])
		self.n = len(self.age)
		self.fx = [0]*self.n

	def rescale(self):
		#rescale the features
		self.age = [(age - statistics.mean(self.age))/statistics.stdev(self.age) for age in self.age]
		self.weight = [(weight - statistics.mean(self.weight)) / statistics.stdev(self.weight) for weight in self.weight]
	def f_x(self):
		#calculate prediction based on current weights
		w0, w1, w2 = self.weights[0], self.weights[1], self.weights[2]
		for i in range(len(self.age)):
			self.fx[i] = w0 + w1 * self.age[i] + w2 * self.weight[i]
		return self.fx

	def gradientDescent(self, new_alpha, new_num, output):
		file = open(output, 'w')
		#write gradient descent results for nine alpha's
		for alpha in self.alpha:
			self.weights = [0, 0, 0]
			smallest_loss = float('inf')
			for j in range(self.num_iter):
				fx = self.f_x()
				sum = [0, 0, 0, 0]
				for i in range(len(self.age)):
					sum[0] += (fx[i] - self.height[i])
					sum[1] += (fx[i] - self.height[i])*self.age[i]
					sum[2] += (fx[i] - self.height[i])*self.weight[i]
					sum[3] += (fx[i] - self.height[i])**2
				self.weights[0] -= (1/self.n) * alpha * sum[0]
				self.weights[1] -= (1 / self.n) * alpha * sum[1]
				self.weights[2] -= (1 / self.n) * alpha * sum[2]
				loss = 1/(2*self.n) * sum[3]
				#display convergence
				#print(loss)
				if loss < smallest_loss:
					smallest_loss = loss

			file.write(str(alpha)+"," + str(self.num_iter) + "," + str(round(self.weights[0], 10))+"," + str(round(self.weights[1], 10))+","+str(round(self.weights[2], 10))+"\n")
		#write output for specified alpha and iteration
		self.weights = [0, 0, 0]
		smallest_loss = float('inf')
		for j in range(new_num):
			fx = self.f_x()
			sum = [0, 0, 0, 0]
			for i in range(len(self.age)):
				sum[0] += (fx[i] - self.height[i])
				sum[1] += (fx[i] - self.height[i]) * self.age[i]
				sum[2] += (fx[i] - self.height[i]) * self.weight[i]
				sum[3] += (fx[i] - self.height[i]) ** 2
			self.weights[0] -= (1 / self.n) * new_alpha * sum[0]
			self.weights[1] -= (1 / self.n) * new_alpha * sum[1]
			self.weights[2] -= (1 / self.n) * new_alpha * sum[2]
		file.write(str(new_alpha) + "," + str(new_num) + "," + str(round(self.weights[0], 10)) + "," + str(round(self.weights[1], 10)) + "," + str(
				round(self.weights[2], 10)) + "\n")
		return self.weights
def main():
	input, output = sys.argv[1], sys.argv[2]
	data = pd.read_csv(input, names=['a', 'w', 'h'])
	l = linearRegressor()
	l.read(data)
	l.rescale()
	w = l.gradientDescent(0.5, 1000, output)
	#data = pd.DataFrame({'a': l.age, 'w': l.weight, 'h': l.height})
	#visualize_3d(data, lin_reg_weights=w, feat1='a', feat2='w', labels='h',  xlim=(-2, 2), ylim=(-3, 3), alpha = 0.5)
if __name__ == '__main__':
	main()