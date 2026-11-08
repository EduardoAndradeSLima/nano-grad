from typing import Set, Tuple, Union


class Value:
    """
    Armazena um valor escalar e o gradiente acumulado.
    Rastreia os nós pais no grafo computacional para execução do backpropagation.
    """

    def __init__(
        self,
        data: Union[int, float],
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
