import argparse
import torch
from models import GRU
import numpy as np

def load_gru_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = GRU(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def run_prediction(model, features, device):
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        h = model.init_hidden(1).to(device)
        y, _ = model(x, h)
        return y.squeeze(0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Predict tongue landmarks using a trained GRU model.")
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input .npy file with features')
    parser.add_argument('--output', required=True, help='File to save predicted landmarks')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index (default: 0)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = load_gru_checkpoint(args.model, device)
    features = np.load(args.input)
    prediction = run_prediction(model, features, device)
    np.save(args.output, prediction)

    print(f"Prediction saved: {args.output}")
    arr = np.load(args.output)
    np.set_printoptions(threshold=np.inf, linewidth=200)
    print("Predicted landmarks array:")
    print(arr)
    print("Array shape:", arr.shape)

if __name__ == "__main__":
    main()
