| Jogador | Partidas | Média | Mediana | Melhor | % tabuleiro (média / máx) | Vitórias | Como terminou |
|---|---:|---:|---:|---:|---:|---:|---|
| Eu (humano) | 20 | 16.40 | 14 | 34 | 21.6% / 41.1% | 0.0% | obstáculo 85%, parede 15% |
| PPO + segurança | 1000 | 43.38 ± 0.82 | 46 | 67 | 51.5% / 77.8% | 0.0% | tempo 88%, corpo 6%, obstáculo 3%, parede 2% |
| PPO, tempo esgotado = -1 | 1000 | 45.50 ± 0.78 | 48 | 69 | 53.9% / 80.0% | 0.0% | tempo 85%, corpo 8%, obstáculo 5%, parede 2% |
| BFS + fim de jogo | 1000 | 50.79 ± 0.74 | 53 | 72 | 59.8% / 83.3% | 0.1% | tempo 84%, obstáculo 8%, corpo 6%, parede 1%, vitória 0% |

± = 2 erros padrão (intervalo de ~95% para a média). Agentes: as mesmas sementes oficiais. Humano: 20 partidas oficiais da Fase 2 (outras sementes).
