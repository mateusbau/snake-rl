| Jogador | Partidas | Média | Mediana | Melhor | % tabuleiro (média / máx) | Vitórias | Como terminou |
|---|---:|---:|---:|---:|---:|---:|---|
| Aleatório | 1000 | 0.94 ± 0.06 | 1 | 6 | 4.4% / 10.0% | 0.0% | parede 89%, obstáculo 6%, corpo 5% |
| Aleatório seguro | 1000 | 7.53 ± 0.22 | 7 | 21 | 11.7% / 26.7% | 0.0% | corpo 52%, parede 33%, obstáculo 15%, tempo 0% |
| Eu (humano) | 20 | 16.40 | 14 | 34 | 21.6% / 41.1% | 0.0% | obstáculo 85%, parede 15% |
| Guloso | 1000 | 18.18 ± 0.42 | 18 | 40 | 23.5% / 47.8% | 0.0% | corpo 50%, obstáculo 41%, parede 9% |
| MaskablePPO | 1000 | 20.26 ± 0.46 | 20 | 45 | 25.9% / 53.3% | 0.0% | corpo 69%, obstáculo 27%, parede 4% |
| MaskablePPO, mais exploração | 1000 | 31.67 ± 0.64 | 32 | 58 | 38.5% / 67.8% | 0.0% | corpo 55%, obstáculo 36%, parede 7%, tempo 2% |
| DQN, 18 números | 1000 | 41.00 ± 0.64 | 41 | 69 | 48.9% / 80.0% | 0.0% | corpo 40%, obstáculo 30%, tempo 18%, parede 12% |
| PPO + currículo | 1000 | 45.84 ± 0.74 | 48 | 67 | 54.3% / 77.8% | 0.0% | tempo 86%, obstáculo 7%, corpo 6%, parede 1% |
| PPO | 1000 | 46.59 ± 0.74 | 49 | 67 | 55.1% / 77.8% | 0.0% | tempo 83%, corpo 10%, obstáculo 4%, parede 3% |
| BFS + flood fill | 1000 | 50.79 ± 0.74 | 53 | 71 | 59.8% / 82.2% | 0.0% | tempo 84%, obstáculo 8%, corpo 6%, parede 1% |

± = 2 erros padrão (intervalo de ~95% para a média). Agentes: as mesmas sementes oficiais. Humano: 20 partidas oficiais da Fase 2 (outras sementes).
