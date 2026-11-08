# nano-grad

Uma engine de diferenciação automática escalar (*autograd*) e redes neurais construída do zero em Python puro, sem nenhuma dependência externa.

Inspirada no [micrograd](https://github.com/karpathy/micrograd) de Andrej Karpathy, o objetivo desta biblioteca é expor a mecânica fundamental do backpropagation e do treinamento de redes neurais com a menor quantidade de abstrações possível.

---

## Como Funciona

O núcleo da biblioteca é a classe `Value`, que encapsula um valor escalar flutuante (`float`) e rastreia o grafo computacional direcionado acíclico (DAG) gerado por operações aritméticas:

```python
from nanograd.engine import Value

# Criação de nós escalares
x = Value(2.0, label="x")
w = Value(-3.0, label="w")
b = Value(1.0, label="b")

# Forward pass: constrói o grafo dinamicamente
y = x * w + b

# Backward pass: propaga os gradientes aplicando a Regra da Cadeia
y.backward()

print(f"y: {y.data}")      # -5.0
print(f"dy/dx: {x.grad}")   # -3.0 (peso w)
print(f"dy/dw: {w.grad}")   #  2.0 (entrada x)
print(f"dy/db: {b.grad}")   #  1.0
```

---

## Estrutura do Repositório

```text
nano-grad/
├── nanograd/
│   ├── __init__.py     # Exportações públicas da biblioteca
│   ├── engine.py       # Classe Value e lógica do Autograd (DAG + backward)
│   └── nn.py           # Módulos neurais (Neuron, Layer, MLP)
├── tests/
│   ├── test_engine.py  # Testes unitários do autograd contra derivadas analíticas
│   └── test_nn.py      # Testes de forward pass e gradientes dos módulos neurais
├── pyproject.toml      # Metadados do pacote
└── README.md
```

---

## Executando os Testes

Os testes utilizam o módulo padrão `unittest` do Python, sem necessidade de instalar dependências extras:

```bash
python -m unittest discover tests
```

Se preferir usar `pytest`:

```bash
pytest
```

---

## Licença

MIT
