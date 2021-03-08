import numpy as np

'''
N -> number of input vectors
m -> number of inputs
n -> number of neurons
'''
class Pcp():
    eta = 1
    iterations = 10
    def __init__(self, inputs, targets):
        self.N = np.shape(inputs)[0]
        self.m = np.shape(inputs)[1]
        self.n = 1
        self.inputs = np.concatenate((inputs, -np.ones((self.N, 1))), axis=1)
        self.targets = np.transpose(targets)
        self.weights = np.random.rand(self.m+1, self.n)*0.1 - 0.05

    def forward(self, inputs):
        self.activations = np.dot(inputs, self.weights)
        self.activations = np.where(self.activations>0, 1, 0)

    def set_weights(self):
        self.weights -= self.eta*np.dot(np.transpose(self.inputs), self.activations-self.targets)     

    def train(self):
        for i in range(self.iterations):
            self.forward(self.inputs)
            self.set_weights()
            print(f"\n{i}: Weights: \n{self.weights} \n Activations: \n{self.activations}")
    
    def test(self, testing):
        testing = np.concatenate((testing, -np.ones((self.N, 1))), axis=1)
        self.forward(testing)
        print(f"Result: \n {self.activations}")

inputs = np.array([[1,1], [0,1], [1,0], [0,0]])

testing = np.array([[1,1], [0,0], [0,0], [1,1]])

targets = np.array([[1,1,1,0]])

model = Pcp(inputs, targets)

print(model.weights)

model.train()

model.test(testing)












