"""Tabuleiros prontos, por nome (usados pelos scripts com --board).

- full: o jogo oficial (9×10, 5 frutas, obstáculos). É o padrão de tudo.
- mini: tabuleiro de TREINO da Fase 4 (5×5, 1 fruta, sem obstáculos), pequeno o
        bastante para uma tabela de Q-learning dar conta.
"""

from snake.env import RIGHT, GameConfig

BOARDS: dict[str, GameConfig] = {
    "full": GameConfig(),
    "mini": GameConfig(
        height=5,
        width=5,
        num_fruits=1,
        # Cobra de 3 células na linha do meio, olhando para a direita; fruta inicial à frente.
        initial_snake=((2, 2), (2, 1), (2, 0)),
        initial_direction=RIGHT,
        initial_fruits=((2, 4),),
        max_steps_without_fruit=50,  # 2× o número de células
        spawn_obstacles=False,
    ),
}
