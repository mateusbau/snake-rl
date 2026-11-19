| Jogador | Partidas | Média | Mediana | Melhor | % tabuleiro (média / máx) | Vitórias | Como terminou |
|---|---:|---:|---:|---:|---:|---:|---|
| Eu (humano) | 20 | 16.40 | 14 | 34 | 21.6% / 41.1% | 0.0% | obstáculo 85%, parede 15% |
| Guloso | 1000 | 18.18 ± 0.42 | 18 | 40 | 23.5% / 47.8% | 0.0% | corpo 50%, obstáculo 41%, parede 9% |
| DQN, 18 números | 1000 | 41.00 ± 0.64 | 41 | 69 | 48.9% / 80.0% | 0.0% | corpo 40%, obstáculo 30%, tempo 18%, parede 12% |
| PPO + segurança | 1000 | 43.38 ± 0.82 | 46 | 67 | 51.5% / 77.8% | 0.0% | tempo 88%, corpo 6%, obstáculo 3%, parede 2% |
| PPO, tempo esgotado = -1 | 1000 | 45.50 ± 0.78 | 48 | 69 | 53.9% / 80.0% | 0.0% | tempo 85%, corpo 8%, obstáculo 5%, parede 2% |
| PPO | 1000 | 46.59 ± 0.74 | 49 | 67 | 55.1% / 77.8% | 0.0% | tempo 83%, corpo 10%, obstáculo 4%, parede 3% |
| BFS + flood fill | 1000 | 50.79 ± 0.74 | 53 | 71 | 59.8% / 82.2% | 0.0% | tempo 84%, obstáculo 8%, corpo 6%, parede 1% |
| BFS + fim de jogo | 1000 | 50.79 ± 0.74 | 53 | 72 | 59.8% / 83.3% | 0.1% | tempo 84%, obstáculo 8%, corpo 6%, parede 1%, vitória 0% |
| PPO + caça + fim | 1000 | 54.91 ± 0.76 | 59 | 69 | 64.3% / 80.0% | 0.0% | tempo 80%, corpo 12%, obstáculo 5%, parede 3% |
| BFS + caça + fim | 1000 | 55.35 ± 0.80 | 59 | 72 | 64.8% / 83.3% | 0.1% | tempo 84%, obstáculo 8%, corpo 7%, parede 1%, vitória 0% |

± = 2 erros padrão (intervalo de ~95% para a média). Agentes: as mesmas sementes oficiais. Humano: 20 partidas oficiais da Fase 2 (outras sementes).
