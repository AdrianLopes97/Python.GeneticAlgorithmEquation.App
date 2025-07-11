# Algoritmo Genético para Solução de Equação

Este projeto utiliza um Algoritmo Genético (AG) para encontrar uma solução de números inteiros para a equação:

**5*x + y² + w + z³ = 185**

O script em Python simula os princípios da evolução natural para convergir para uma combinação de valores `(x, y, w, z)` que satisfaça a equação.

## Conceitos do Algoritmo Genético Aplicados

O código implementa os seguintes conceitos fundamentais de algoritmos genéticos:

-   **Cromossomo**: Representa uma solução individual (um candidato). Neste projeto, um cromossomo é um dicionário com os quatro genes: `{ 'x': valor, 'y': valor, 'w': valor, 'z': valor }`.

-   **População**: Um conjunto de cromossomos. O algoritmo trabalha com uma população de soluções candidatas a cada geração.

-   **Função de Aptidão (Fitness)**: Uma função que avalia quão "boa" é uma solução. O objetivo é maximizar essa função. Aqui, a aptidão é calculada com base na proximidade do resultado da equação ao valor alvo (185). Uma aptidão de `1.0` significa que a solução exata foi encontrada.
    ```
    aptidao = 1 / (1 + abs(185 - resultado_da_equacao))
    ```

-   **Seleção**: O processo de escolher os indivíduos mais aptos da população para serem "pais" da próxima geração. Este projeto utiliza o método de **Seleção por Roleta**, onde os indivíduos com maior aptidão têm maior probabilidade de serem escolhidos.

-   **Cruzamento (Crossover)**: Combina o material genético de dois pais para criar um "filho" (uma nova solução). Foi implementado o **crossover de um ponto**, onde um ponto de corte é escolhido aleatoriamente, e o filho herda os genes do primeiro pai até esse ponto e do segundo pai após esse ponto.

-   **Mutação**: Introduz pequenas e aleatórias alterações nos genes de um cromossomo. Isso garante a diversidade genética na população, evitando que o algoritmo fique preso em soluções sub-ótimas.

## Como Executar

1.  **Pré-requisitos**: Certifique-se de ter o Python 3 instalado.
2.  **Execute o script**: Abra um terminal na pasta do projeto e execute o seguinte comando:
    ```bash
    python main.py
    ```

O script irá imprimir o melhor indivíduo de cada geração, mostrando a evolução da solução até encontrar o resultado.

### Exemplo de Saída

```
Melhores indivíduos por geração:
Geração 0: {'x': 36, 'y': 14, 'w': 37, 'z': 4} -> Resultado: 477, Fitness: 0.0034
Geração 1: {'x': 36, 'y': 0, 'w': 27, 'z': 5} -> Resultado: 332, Fitness: 0.0068
...
Geração 15: {'x': 15, 'y': 2, 'w': 41, 'z': 4} -> Resultado: 184, Fitness: 0.5000
Geração 16: {'x': 15, 'y': 2, 'w': 42, 'z': 4} -> Resultado: 185, Fitness: 1.0000
```

## Configuração

Você pode ajustar os parâmetros do algoritmo genético diretamente no arquivo `main.py`:

-   `POPULATION_SIZE`: O número de indivíduos em cada geração.
-   `NUM_GENERATIONS`: O número máximo de gerações que o algoritmo executará.
-   `MUTATION_RATE`: A probabilidade (entre 0 e 1) de um gene sofrer mutação.
-   `TARGET_RESULT`: O valor alvo da equação (neste caso, 185).
-   `SEARCH_RANGES`: Os intervalos de valores permitidos para cada variável (`x`, `y`, `w`, `z`).
