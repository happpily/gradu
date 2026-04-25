import argparse

from train.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN-based wireless power allocation with thesis-ready evaluation metrics."
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    parser.add_argument("--num-links", type=int, default=10, help="Number of D2D links")
    parser.add_argument("--area-size", type=float, default=100.0, help="Simulation area size")
    parser.add_argument("--d-min", type=float, default=2.0, help="Minimum TX-RX distance")
    parser.add_argument("--d-max", type=float, default=10.0, help="Maximum TX-RX distance")
    parser.add_argument("--pathloss-exp", type=float, default=2.0, help="Pathloss exponent")
    parser.add_argument("--noise-power", type=float, default=1e-3, help="Noise power")
    parser.add_argument("--p-max", type=float, default=1.0, help="Maximum transmit power")
    parser.add_argument("--rate-threshold", type=float, default=1.0, help="QoS rate threshold")

    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--val-size", type=int, default=100, help="Validation set size")
    parser.add_argument("--test-size", type=int, default=200, help="Test set size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--qos-lambda", type=float, default=0.2, help="QoS penalty weight")
    parser.add_argument("--dropout", type=float, default=0.1, help="GAT dropout")
    parser.add_argument("--val-every", type=int, default=20, help="Validation frequency")
    parser.add_argument("--log-every", type=int, default=20, help="Logging frequency")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
