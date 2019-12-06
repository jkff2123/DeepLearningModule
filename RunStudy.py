import sys, os, json
from Classes import DataHandler
from Classes.Library import np
from Classes.Neural import NeuralNet

json_data = open(os.path.dirname(__file__) + "/config/Config.json").read()
config = json.loads(json_data)

x = DataHandler.loaddata(config["study_file_path"]["samplefile_path"], np.float)
t = DataHandler.loaddata(config["study_file_path"]["trainingfile_path"], np.int32)

for i in range(x.shape[0]):
    x[i,] = (x[i,] - np.min(x[i,])) / (np.max(x[i,]) - np.min(x[i,]))

nodes_size = [x.shape[1]]
for i in config["config"]["node_numbers"]:
    nodes_size.append(i)

NN = NeuralNet(nodes_size, load = config["config"]["load_exist_weight_data"], weight = config["config"]["weight"])
NN.neuralstudy(x, t, config["config"]["batch_size"]
, config["config"]["iteration_number"]
, config["config"]["ln_rate"])