# Fase 0 — Preparação

## O que foi feito

- Estrutura de pastas do projeto (`snake/`, `agents/`, `tests/`, `docs/`, `scripts/`, `experiments/`).
- Ambiente virtual `.venv` com Python 3.12.10 e `pytest`.
- `pyproject.toml` configurando o pytest; `requirements.txt`; `.gitignore`; README com "como rodar".
- Teste de fumaça (`tests/test_setup.py`) e repositório Git publicado no GitHub.

## Conceitos novos (com equivalentes em JavaScript)

| Python | JavaScript | Para que serve |
|---|---|---|
| `.venv/` | `node_modules/` | Bibliotecas instaladas só para este projeto |
| `requirements.txt` | `dependencies` do `package.json` | Lista do que instalar |
| `pyproject.toml` | `package.json` (parte de config) | Metadados e configuração de ferramentas |
| `pytest` | Jest | Roda os testes |
| `assert x == y` | `expect(x).toBe(y)` | Verifica um resultado |
| `__init__.py` | `index.js` de uma pasta | Transforma a pasta num pacote importável |

**Exemplo resolvido — como o pytest acha os testes:**

1. O `pyproject.toml` diz `testpaths = ["tests"]`, então o pytest olha só a pasta `tests/`.
2. Lá dentro, ele abre todo arquivo `test_*.py`.
3. Em cada arquivo, roda toda função `test_*`.
4. Se algum `assert` for falso, aquele teste aparece como `FAILED`; senão, `PASSED`.

**Por que a `.venv` não vai para o Git:** ela tem ~2.000 arquivos gerados pela
instalação e é recriada com `pip install -r requirements.txt` — igual ao
`node_modules`, que também nunca se versiona.

## Como rodar

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

## Resultados

`pytest`: 2 testes passando (`test_python_version`, `test_packages_importable`).

## O que deu errado e como foi resolvido

1. **Arquivos vazios.** Todos os arquivos criados à mão estavam com 0 bytes
   (provavelmente criados mas não salvos no editor). Preenchidos; a especificação
   foi salva de novo.
2. **Pastas aninhadas.** `agents/`, `docs/` e `tests/` tinham ido parar dentro de
   `snake/agents/docs/...`. Movidas para a raiz.
3. **`.venv` commitada.** Com o `.gitignore` vazio, o primeiro commit levou a
   `.venv` inteira. Como ainda não tinha sido publicado, o commit foi desfeito e
   refeito com o `.gitignore` correto. Lição: escrever o `.gitignore` **antes** do
   primeiro `git add`.
