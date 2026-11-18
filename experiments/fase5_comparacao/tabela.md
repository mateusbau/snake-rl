| Jogador | Partidas | Média | Mediana | Melhor | % tabuleiro (média / máx) | Vitórias | Como terminou |
|---|---:|---:|---:|---:|---:|---:|---|
| Aleatório | 1000 | 0.94 ± 0.06 | 1 | 6 | 4.4% / 10.0% | 0.0% | parede 89%, obstáculo 6%, corpo 5% |
| Aleatório seguro | 1000 | 7.53 ± 0.22 | 7 | 21 | 11.7% / 26.7% | 0.0% | corpo 52%, parede 33%, obstáculo 15%, tempo 0% |
| Eu (humano) | 20 | 16.40 | 14 | 34 | 21.6% / 41.1% | 0.0% | obstáculo 85%, parede 15% |
| Guloso | 1000 | 18.18 ± 0.42 | 18 | 40 | 23.5% / 47.8% | 0.0% | corpo 50%, obstáculo 41%, parede 9% |
| DQN, 11 números | 1000 | 18.43 ± 0.44 | 18 | 41 | 23.8% / 48.9% | 0.0% | corpo 62%, obstáculo 24%, parede 15%, tempo 0% |
| DQN, tabuleiro (CNN) | 1000 | 24.38 ± 0.42 | 25 | 48 | 30.4% / 56.7% | 0.0% | corpo 54%, obstáculo 32%, parede 14% |
| DQN, 18 números | 1000 | 41.00 ± 0.64 | 41 | 69 | 48.9% / 80.0% | 0.0% | corpo 40%, obstáculo 30%, tempo 18%, parede 12% |
| BFS + flood fill | 1000 | 50.79 ± 0.74 | 53 | 71 | 59.8% / 82.2% | 0.0% | tempo 84%, obstáculo 8%, corpo 6%, parede 1% |

± = 2 erros padrão (intervalo de ~95% para a média). Agentes: as mesmas sementes oficiais. Humano: 20 partidas oficiais da Fase 2 (outras sementes).
