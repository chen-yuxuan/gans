from gans.datasets import mnist


def test_mnist():
    dataset = mnist(root="./datasets", train=True, download=False)
    assert len(dataset) == 60000

    