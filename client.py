import flwr as fl
import sys
from model import create_model
from dataset import load_datasets

# Get client ID from terminal argument
if len(sys.argv) < 2:
    print("Please run like: python client.py 0")
    exit()

cid = sys.argv[1]

client_data, test_data = load_datasets()


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid):
        self.cid = cid
        self.model = create_model()
        self.x, self.y = client_data[int(cid)]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"Client {self.cid} training...")
        self.model.set_weights(parameters)
        self.model.fit(self.x, self.y, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x, self.y, verbose=0)
        return loss, len(self.x), {"accuracy": acc}


def client_fn(cid):
    return FlowerClient(cid)


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client_fn(cid),
)

#run as python client.py 0....