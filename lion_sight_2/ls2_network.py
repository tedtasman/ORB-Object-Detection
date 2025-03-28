import torch

class LS2Network:

    def __init__(self, net_path):
        """
        Initialize the LS2Network with the specified model path.
        """
        self.net_path = net_path
        self.net = self.load_net(net_path)

        if torch.cuda.is_available():
            self.net = self.net.cuda()
        else:
            print("CUDA is not available. Using CPU.")
            self.net = self.net.cpu()
    

    def load_net(self, net_path):
        """
        Load the neural network model from the specified path.
        """
        if net_path is None:
            return None
        self.net = torch.load(net_path)
        self.net.eval()
        print(f"Loaded network from {net_path}")

    
    # TODO: Crop down on point of interest, and run the network on that