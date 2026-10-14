import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

def save_case(name, inputs, expected):

    path = OUT / f"{name}.txt"
    lines = [name]

    for label, arr in inputs:
        arr = np.array(arr, dtype=np.float32)
        lines.append(label)
        lines.append(" ".join(str(n) for n in arr.shape))
        lines.append(" ".join(str(n) for n in arr.ravel(order="C")))

    expected = np.asarray(expected, dtype=np.float32)
    lines.append("expected")
    lines.append(" ".join(str(n) for n in expected.shape))
    lines.append(" ".join(f"{v:.8g}" for v in expected.ravel(order="C")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def binary(name, op, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    save_case(name, [("input_a", a), ("input_b", b)], op(a, b))

def linear():

    torch.manual_seed(42)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    layer = nn.Linear(3, 4)
    y = layer(x)

    grad_out = torch.tensor([[0.1, -0.2, 0.3, 0.0], [0.2, 0.1, -0.4, 0.5]], dtype=torch.float32)
    y.backward(grad_out)

    save_case("linear", [
        ("input", x.detach().numpy()),
        ("weight", layer.weight.detach().numpy()),
        ("bias", layer.bias.detach().numpy()),
        ("grad_out", grad_out.numpy()),
        ("output", y.detach().numpy()),
        ("grad_in", x.grad.numpy()),
        ("weight_grad", layer.weight.grad.numpy()),
        ("bias_grad", layer.bias.grad.numpy()),
    ], y.detach().numpy())


def embedding():

    torch.manual_seed(42)
    x = torch.tensor([1, 2, 0], dtype=torch.long)
    layer = nn.Embedding(3, 4)
    y = layer(x)

    grad_out = torch.tensor([[0.1, -0.2, 0.3, 0.0], [0.2, 0.1, -0.4, 0.5], [0.3, 0.2, 0.0, -0.2]], dtype=torch.float32)
    y.backward(grad_out)

    save_case("embedding", [
        ("input", x.detach().numpy()),
        ("weight", layer.weight.detach().numpy()),
        ("grad_out", grad_out.numpy()),
        ("output", y.detach().numpy()),
        ("weight_grad", layer.weight.grad.numpy()),
    ], y.detach().numpy())


def layer_norm():

    torch.manual_seed(42)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    layer = nn.LayerNorm(3)
    y = layer(x)

    grad_out = torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.4]], dtype=torch.float32)
    y.backward(grad_out)

    save_case("layer_norm", [
        ("input", x.detach().numpy()),
        ("gamma", layer.weight.detach().numpy()),
        ("beta", layer.bias.detach().numpy()),
        ("grad_out", grad_out.numpy()),
        ("output", y.detach().numpy()),
        ("grad_in", x.grad.numpy()),
        ("gamma_grad", layer.weight.grad.numpy()),
        ("beta_grad", layer.bias.grad.numpy()),
    ], y.detach().numpy())


def gelu():

    torch.manual_seed(42)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    layer = nn.GELU(approximate="tanh")
    y = layer(x)

    grad_out = torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.4]], dtype=torch.float32)
    y.backward(grad_out)

    save_case("gelu", [
        ("input", x.detach().numpy()),
        ("grad_out", grad_out.numpy()),
        ("output", y.detach().numpy()),
        ("grad_in", x.grad.numpy()),
    ], y.detach().numpy())


def softmax():

    torch.manual_seed(42)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    layer = nn.Softmax(-1)
    y = layer(x)

    grad_out = torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.4]], dtype=torch.float32)
    y.backward(grad_out)

    save_case("softmax", [
        ("input", x.detach().numpy()),
        ("axis", np.array([-1], dtype=np.float32)),
        ("grad_out", grad_out.numpy()),
        ("output", y.detach().numpy()),
        ("grad_in", x.grad.numpy()),
    ], y.detach().numpy())

def main():

    OUT.mkdir(parents=True, exist_ok=True)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    same = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.float32)
    row = np.array([10, 20, 30], dtype=np.float32)
    col = np.array([[10], [20]], dtype=np.float32)

    for op_name, op in (("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)):
        binary(f"{op_name}_same", op, a, same)
        binary(f"{op_name}_broadcast_axis", op, a, row)
        binary(f"{op_name}_broadcast_one", op, a, col)

    left = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    right = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    binary("matmul_small", np.matmul, left, right)

    rng = np.random.default_rng(42)
    binary("matmul_out", np.matmul, rng.standard_normal((64, 32), dtype=np.float32), rng.standard_normal((32, 5256), dtype=np.float32))

    batch_a = rng.standard_normal((2, 2, 3), dtype=np.float32)
    batch_b = rng.standard_normal((2, 3, 2), dtype=np.float32)
    binary("batch_matmul_small", np.matmul, batch_a, batch_b)

    linear()
    embedding()
    layer_norm()
    gelu()
    softmax()

if __name__ == "__main__":
    main()