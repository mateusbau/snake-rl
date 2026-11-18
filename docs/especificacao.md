# DOCUMENTAÇÃO TÉCNICA — SNAKE COM OBSTÁCULOS DINÂMICOS
## Especificação do ambiente para desenvolvimento de um agente de Machine Learning

---

# 1. Visão geral do projeto

O projeto consiste no desenvolvimento de um agente de Machine Learning capaz de jogar uma versão modificada do clássico jogo Snake (jogo da cobrinha).

O objetivo principal é desenvolver um agente que consiga navegar pelo tabuleiro, coletar frutas, evitar colisões e adaptar sua estratégia à medida que o ambiente se torna progressivamente mais complexo.

A principal particularidade desta versão é a presença de **obstáculos dinâmicos**, que são adicionados ao mapa conforme o jogador coleta frutas.

Diferentemente da versão tradicional do Snake, na qual o ambiente geralmente permanece estático, neste modo o espaço navegável é progressivamente reduzido pela geração de obstáculos permanentes.

O ambiente possui regras específicas para a geração desses obstáculos, incluindo restrições de proximidade e uma região de segurança à frente da cobra.

A documentação abaixo estabelece as regras que devem ser respeitadas na implementação do ambiente.

# 2. Configuração do ambiente

## 2.1. Tabuleiro

O ambiente utiliza uma grade bidimensional composta por células quadradas.

Configuração inicial:

- Altura: 9 células.
- Largura: 10 células.
- Total: 90 células.
- Sistema de coordenadas: linhas e colunas, com índices iniciando em zero.

Cada célula pode representar:
- Espaço vazio.
- Segmento da cobra.
- Fruta.
- Obstáculo.

Uma célula não pode conter simultaneamente dois elementos diferentes.

## 2.2. Estado inicial da cobra

A cobra começa a partida com:
- Comprimento inicial de 3 células.
- Uma cabeça.
- Dois segmentos de corpo.
- Uma direção inicial de movimento.

A posição inicial e a direção devem ser configuráveis na implementação, pois ainda não foram estabelecidas como valores fixos.

## 2.3. Frutas

O mapa deve possuir sempre 5 frutas simultaneamente.

Regras:
1. As frutas são distribuídas aleatoriamente em posições válidas do tabuleiro.
2. Quando a cobra coleta uma fruta, uma nova fruta deve aparecer imediatamente.
3. A quantidade de frutas disponíveis deve permanecer em 5 durante toda a partida, enquanto houver posições disponíveis.
4. Uma fruta não pode aparecer em uma célula ocupada pela cobra, por um obstáculo ou por outra fruta.
5. Frutas podem aparecer próximas à cobra e aos obstáculos, desde que não ocupem suas células.

A distribuição das novas frutas deve ser aleatória e uniforme entre as células elegíveis.

# 3. Sistema de obstáculos dinâmicos

Esta é a principal característica diferencial do modo de jogo.

## 3.1. Regra de geração

Um novo obstáculo é gerado a cada quantidade ímpar de frutas coletadas.

A geração começa imediatamente após a primeira fruta coletada.

Portanto:

| Frutas coletadas | Obstáculos gerados acumulados |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 1 |
| 3 | 2 |
| 4 | 2 |
| 5 | 3 |
| 6 | 3 |
| 7 | 4 |
| 9 | 5 |
| 11 | 6 |

Regra matemática:

Um obstáculo deve ser gerado quando:

`quantidade_de_frutas_coletadas % 2 == 1`

A cada marco ímpar, deve ser realizada uma tentativa de geração de exatamente um novo obstáculo.

## 3.2. Permanência dos obstáculos

Os obstáculos possuem as seguintes propriedades:

- São estáticos.
- Permanecem na mesma posição após sua geração.
- Não desaparecem.
- Não mudam de posição.
- Não são removidos quando a cobra passa por outras regiões do mapa.
- Permanecem até o encerramento da partida.

A quantidade de obstáculos não possui um limite numérico predefinido.

A geração continua enquanto existirem posições válidas para a inserção de novos obstáculos.

## 3.3. Distribuição dos obstáculos

A posição de cada novo obstáculo deve ser escolhida aleatoriamente entre todas as células que atendam aos critérios de posicionamento.

A distribuição deve ser uniforme, ou seja, todas as posições válidas devem possuir a mesma probabilidade de seleção.

Não deve existir preferência artificial por:
- Bordas.
- Cantos.
- Centro do mapa.
- Proximidade ou distância das frutas.
- Proximidade ou distância do corpo da cobra.

A única influência sobre a seleção deve ser o conjunto de restrições descrito nesta documentação.

# 4. Restrições para geração de obstáculos

A geração de obstáculos deve respeitar obrigatoriamente todas as condições abaixo.

Uma posição somente será considerada válida se atender simultaneamente a todos os critérios.

## 4.1. Restrição de ocupação

Um obstáculo não pode nascer:
- Sobre a cabeça da cobra.
- Sobre qualquer segmento do corpo da cobra.
- Sobre uma fruta.
- Sobre outro obstáculo.

## 4.2. Restrição de adjacência entre obstáculos

Um novo obstáculo não pode nascer em nenhuma das oito células vizinhas de um obstáculo já existente.

Isso inclui:
- Célula imediatamente acima.
- Célula imediatamente abaixo.
- Célula imediatamente à esquerda.
- Célula imediatamente à direita.
- As quatro células diagonais.

Portanto, dois obstáculos não podem ocupar células adjacentes horizontalmente, verticalmente ou diagonalmente.

Essa regra considera exclusivamente a proximidade entre obstáculos.

**Importante:** essa restrição não se aplica à proximidade entre obstáculos e frutas ou entre obstáculos e segmentos do corpo da cobra.

Um obstáculo pode nascer imediatamente ao lado de uma fruta ou de um segmento do corpo, desde que não ocupe a mesma célula e respeite a região de segurança da cabeça.

## 4.3. Região de segurança à frente da cobra

Para evitar que obstáculos apareçam inesperadamente diante da cabeça da cobra, existe uma região de segurança dinâmica.

A região é determinada pela posição e pela direção atual da cabeça.

**Regra:** nenhum obstáculo pode nascer nas três células consecutivas imediatamente à frente da cabeça da cobra, seguindo a direção atual de movimento.

Exemplo:

Se a cobra estiver se movimentando para a direita, as três células imediatamente à direita da cabeça estarão temporariamente protegidas contra a geração de obstáculos.

Se estiver se movimentando para cima, as três células imediatamente acima estarão protegidas.

Essa regra acompanha a direção atual da cobra, não sendo uma região fixa do mapa.

A região de segurança possui exatamente três células de profundidade, sem considerar uma área diagonal ou um cone de proteção.

Caso parte dessa região ultrapasse os limites do tabuleiro, devem ser consideradas somente as células existentes dentro do mapa.

## 4.4. Proximidade das bordas

Não existe restrição especial que impeça a geração de obstáculos próximos às bordas.

Uma célula próxima à borda pode receber um obstáculo, desde que respeite todas as demais condições.

## 4.5. Proibição de becos sem saída

> **Decisão do projeto (2026-11-17), não presente na versão original desta especificação.** Esta regra **diverge do Snake original do Google**, que permite esses becos. Motivo: como a cobra não pode dar meia-volta, entrar numa célula com uma única saída é morte garantida. Em simulações com a regra antiga, 36,7% das partidas tiveram pelo menos um beco e 24,5% dos obstáculos nasceram criando um.

Um beco sem saída é uma célula livre com apenas **uma** vizinha ortogonal livre. Exemplos (■ = obstáculo, x = beco, a linha/coluna externa é a parede):

```
┌──────────        ┌──────────        ┌──────────
│ x ■ . .          │ ■ x ■ .          │ . . . .
│ . . . .          │ . . . .          │ x ■ . .
                                      │ . . . .
```

Um espaço de uma célula entre dois obstáculos **no meio do mapa** continua permitido, porque a célula tem saída pelos dois lados (é um corredor, não um beco):

```
│ . . . . .
│ . ■ . ■ .   <- a célula entre os dois tem saída para cima e para baixo: OK
│ . . . . .
```

**Regra:** uma célula candidata P só é válida para um novo obstáculo se, depois de colocar o obstáculo em P, toda célula livre do tabuleiro continuar com pelo menos 2 vizinhas ortogonais livres (cima, baixo, esquerda, direita).

Definições:
- Contam como **bloqueio** apenas o que é permanente: as bordas do tabuleiro e os obstáculos.
- Cobra e frutas **não** contam como bloqueio (a cobra se move e as frutas são comidas). Para esta regra, as células delas são consideradas livres.
- Na prática, só as vizinhas ortogonais de P mudam de contagem. Basta checar: para cada vizinha ortogonal Q de P que esteja dentro do tabuleiro e não seja obstáculo, Q precisa ter pelo menos 2 vizinhas livres sem contar P.
- Consequência: nenhum obstáculo pode nascer encostado (ortogonalmente) numa célula de canto, nem deixar uma célula de espaço entre ele e outro obstáculo quando essa célula estiver encostada na parede.

Todas as outras regras continuam iguais (ocupação, adjacência de 8 direções, região de segurança e sorteio uniforme entre as células que atendem **todas** as restrições). Se nenhuma célula for válida, a tentativa daquele marco é cancelada, como na seção 5.1.

# 5. Procedimento de geração de obstáculos

O processo de geração deve seguir esta sequência:

1. A cobra coleta uma fruta.
2. O contador de frutas coletadas é incrementado.
3. Uma nova fruta é posicionada em uma célula válida.
4. O sistema verifica se a quantidade de frutas coletadas é ímpar.
5. Caso seja ímpar, inicia-se uma tentativa de geração de obstáculo.
6. O sistema percorre as células do tabuleiro e identifica todas as posições elegíveis.
7. As posições que violam qualquer restrição são descartadas.
8. Caso existam posições válidas, uma delas é selecionada aleatoriamente.
9. O obstáculo é inserido permanentemente no mapa.
10. Caso não existam posições válidas, a tentativa daquele marco é cancelada.

## 5.1. Ausência de posições válidas

Caso o sistema não encontre nenhuma posição válida no momento de uma tentativa de geração:

- Nenhum obstáculo será criado naquele marco.
- A partida continuará normalmente.
- A tentativa não deverá causar erro ou interromper a execução do jogo.
- Uma nova tentativa ocorrerá no próximo marco ímpar de frutas coletadas.

A ausência de posições válidas não representa automaticamente uma condição de derrota.

# 6. Mecânica de movimentação e colisões

## 6.1. Movimentação

A cobra se movimenta continuamente pelo tabuleiro, seguindo uma direção.

As direções possíveis são:
- Cima.
- Baixo.
- Esquerda.
- Direita.

A movimentação deve respeitar a estrutura da grade.

A implementação deve separar a lógica de movimentação da lógica de geração de frutas e obstáculos.

## 6.2. Crescimento da cobra

O comprimento inicial é de 3 células.

A regra exata de crescimento após a coleta de frutas ainda não foi especificada e deverá ser definida como parâmetro da implementação, sem ser confundida com uma regra já confirmada do ambiente.

## 6.3. Condições de derrota

A partida termina imediatamente quando ocorre qualquer uma das seguintes situações:

**Colisão com o próprio corpo**
- A cabeça da cobra ocupa uma célula que já contém um segmento do próprio corpo.

**Colisão com obstáculos**
- A cabeça da cobra ocupa uma célula que contém um obstáculo.

**Colisão com os limites do mapa**
- A cabeça da cobra ultrapassa os limites do tabuleiro.

Não existe mecânica de atravessar uma borda e reaparecer na borda oposta.

O contato com os limites do mapa representa derrota.

# 7. Condições especiais do ambiente

## 7.1. Ausência de posições para novas frutas

A reposição de frutas deve respeitar as restrições de ocupação.

Se não existir nenhuma célula disponível para uma nova fruta, a implementação deverá tratar essa situação como um estado especial de encerramento ou impossibilidade de reposição.

O tratamento definitivo desse cenário deverá ser definido durante o desenvolvimento.

## 7.2. Progressão da dificuldade

A dificuldade do ambiente aumenta progressivamente devido à geração permanente de obstáculos.

A cada marco ímpar de frutas coletadas, existe a possibilidade de redução do espaço navegável.

Consequentemente:
- O espaço livre diminui ao longo da partida.
- A quantidade de rotas possíveis pode diminuir.
- A navegação pode exigir planejamento de trajetórias.
- A cobra precisa considerar tanto os obstáculos existentes quanto as futuras possibilidades de geração.

# 8. Representação do estado do ambiente para Machine Learning

A implementação deve manter uma representação estruturada do estado atual do jogo.

Recomenda-se que o estado do ambiente contenha, no mínimo:

| Variável | Descrição |
|---|---|
| snake_body | Lista ordenada das coordenadas de todos os segmentos da cobra |
| snake_head | Coordenada atual da cabeça |
| snake_direction | Direção atual de movimentação |
| fruits | Lista de coordenadas das frutas disponíveis |
| obstacles | Lista de coordenadas de todos os obstáculos |
| fruits_collected | Quantidade acumulada de frutas coletadas |
| obstacles_generated | Quantidade acumulada de obstáculos efetivamente gerados |
| board_width | Largura do tabuleiro |
| board_height | Altura do tabuleiro |
| game_over | Indicador de encerramento da partida |

É importante manter a representação do ambiente independente do algoritmo de Machine Learning utilizado.

A estrutura deve permitir que diferentes agentes consultem o estado do jogo e interajam com ele.

## 8.1. Observações sobre o estado

O agente precisa conseguir identificar:
- Sua posição atual.
- Sua direção.
- A posição do próprio corpo.
- A localização de todas as frutas.
- A localização de todos os obstáculos.
- Os limites do tabuleiro.
- As células que podem representar riscos de colisão.

Também é recomendável disponibilizar uma função que retorne as posições atualmente elegíveis para a geração de obstáculos.

Isso permitirá analisar e testar a mecânica independentemente do agente.

# 9. Separação entre ambiente e agente

Uma decisão arquitetural importante é manter duas estruturas independentes.

**Game Environment**

Responsável por:
- Armazenar o estado do jogo.
- Executar a movimentação da cobra.
- Detectar colisões.
- Controlar a coleta e reposição de frutas.
- Gerar obstáculos.
- Validar as posições de geração.
- Atualizar a pontuação.
- Determinar o encerramento da partida.

**Machine Learning Agent**

Responsável por:
- Receber as informações do ambiente.
- Interpretar o estado atual.
- Selecionar uma ação.
- Definir a direção da cobra.
- Aprender estratégias de navegação.
- Maximizar a coleta de frutas.
- Evitar colisões e situações de aprisionamento.

O agente não deve implementar diretamente as regras internas do jogo.

Ele deve interagir com o ambiente por meio de uma interface bem definida.

# 10. Recomendações para desenvolvimento e testes

Antes de iniciar o treinamento do agente, recomenda-se desenvolver e validar completamente o ambiente.

Os seguintes testes devem ser realizados:

1. Verificar se o tabuleiro possui exatamente 90 células.
2. Verificar se existem 5 frutas no início da partida.
3. Verificar se a cobra começa com 3 segmentos.
4. Verificar se a primeira fruta coletada gera um obstáculo.
5. Verificar se a segunda fruta não gera um obstáculo adicional.
6. Verificar se a terceira fruta gera outro obstáculo.
7. Verificar se os obstáculos permanecem estáticos.
8. Verificar se dois obstáculos nunca ficam adjacentes, inclusive diagonalmente.
9. Verificar se obstáculos nunca aparecem sobre frutas ou sobre a cobra.
10. Verificar se a região de segurança de três células é respeitada.
11. Verificar se obstáculos podem aparecer próximos ao corpo da cobra e às frutas.
12. Verificar se a seleção das posições válidas é uniforme.
13. Verificar o comportamento quando não existem posições válidas para obstáculos.
14. Verificar a reposição das frutas.
15. Verificar todas as condições de derrota.
16. Verificar que, com o tabuleiro sem obstáculos, as células vizinhas ortogonais dos cantos (ex.: (0,1) e (1,0)) são inválidas (seção 4.5).
17. Verificar que, com um obstáculo em (0,0), a célula (0,2) é inválida (a célula (0,1) ficaria presa entre a parede e dois obstáculos).
18. Verificar que, com um obstáculo em (4,4), a célula (4,6) é válida (corredor no meio do mapa).
19. Verificar que a função que lista as posições válidas para obstáculos exclui exatamente as células que criariam beco.
20. Teste de propriedade: em milhares de partidas aleatórias com sementes, depois de cada passo, nenhuma célula livre tem menos de 2 vizinhas livres, e todas as células livres continuam conectadas entre si (busca em largura considerando só paredes e obstáculos como bloqueio).

Recomenda-se também implementar um modo de simulação sem interface gráfica, permitindo executar milhares de partidas e testar o comportamento do ambiente com maior eficiência.

# 11. Diretrizes para a LLM durante o desenvolvimento

Ao utilizar esta documentação como contexto, considere as seguintes orientações:

- Respeite integralmente as regras descritas.
- Não substitua a mecânica de geração de obstáculos pela versão tradicional do Snake.
- Não simplifique ou remova as restrições de posicionamento.
- Não trate a região de segurança como uma área estática.
- Não permita que obstáculos sejam removidos após sua geração.
- Não altere a frequência de geração de obstáculos.
- Mantenha a lógica do ambiente independente do algoritmo de Machine Learning.
- Identifique explicitamente qualquer requisito ainda não especificado antes de assumir um comportamento.
- Priorize modularidade, eficiência, testabilidade e reprodutibilidade.

**O objetivo é construir primeiro um ambiente de jogo confiável e determinístico, com aleatoriedade controlada e reproduzível, para posteriormente desenvolver e treinar um agente capaz de jogar estrategicamente.**

---

# 12. Resumo das regras fundamentais

| Parâmetro | Configuração |
|---|---|
| Tabuleiro | 9 × 10 |
| Total de células | 90 |
| Comprimento inicial da cobra | 3 células |
| Frutas simultâneas | 5 |
| Reposição de frutas | Imediata |
| Primeira geração de obstáculo | 1ª fruta |
| Frequência de geração | A cada quantidade ímpar de frutas |
| Obstáculos por marco | 1 |
| Permanência | Permanente |
| Adjacência entre obstáculos | Proibida nas 8 direções |
| Distância de segurança da cabeça | 3 células à frente |
| Becos sem saída | Proibidos: toda célula livre mantém ≥ 2 vizinhas ortogonais livres (seção 4.5; decisão do projeto, diverge do Google) |
| Distribuição | Aleatória uniforme entre posições válidas |
| Limite de obstáculos | Determinado pela disponibilidade de posições válidas |
| Colisão com o corpo | Derrota |
| Colisão com obstáculos | Derrota |
| Colisão com bordas | Derrota |
| Travessia entre bordas | Não permitida |

**Definição final:** trata-se de uma variante do Snake com ambiente progressivamente obstruído, na qual obstáculos permanentes são adicionados ao tabuleiro a cada quantidade ímpar de frutas coletadas, respeitando restrições espaciais de adjacência, ocupação e segurança de movimentação. O agente deve aprender a navegar em um espaço que se modifica ao longo da partida, exigindo adaptação contínua de suas estratégias.
