"""Teste de fumaça da Fase 0: confirma que o ambiente está montado certo.

O pytest roda automaticamente toda função cujo nome começa com "test_".
O `assert` funciona como um expect(...).toBe(...) do Jest: se a condição
for falsa, o teste falha.
"""

import sys


def test_python_version():
    # O projeto usa Python 3.12 (o 3.14 do sistema não tem todas as
    # bibliotecas que vamos precisar, como o pygame).
    assert sys.version_info[:2] == (3, 12)


def test_packages_importable():
    # Garante que os pacotes do projeto são importáveis a partir da raiz.
    import agents
    import snake

    assert snake is not None and agents is not None
