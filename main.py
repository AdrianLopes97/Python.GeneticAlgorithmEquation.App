import random

# --- Constantes ---
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
TARGET_RESULT = 185

# Intervalos de busca para cada variável
SEARCH_RANGES = {
    'x': (0, 50),
    'y': (0, 50),
    'w': (0, 50),
    'z': (0, 20)
}

class GeneticAlgorithm:
    """
    Uma classe para encapsular a lógica do Algoritmo Genético para resolver a equação:
    5*x + y**2 + w + z**3 = TARGET_RESULT
    """
    def __init__(self, population_size, mutation_rate, num_generations, search_ranges, target):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.search_ranges = search_ranges
        self.target = target
        self.population = []
        self.best_individuals_history = []

    def _create_chromosome(self):
        """Gera um cromossomo aleatório (uma solução em potencial)."""
        return {
            'x': random.randint(*self.search_ranges['x']),
            'y': random.randint(*self.search_ranges['y']),
            'w': random.randint(*self.search_ranges['w']),
            'z': random.randint(*self.search_ranges['z'])
        }

    def _calculate_fitness(self, chromosome):
        """Calcula a aptidão de um cromossomo."""
        result = (5 * chromosome['x'] + 
                  chromosome['y']**2 + 
                  chromosome['w'] + 
                  chromosome['z']**3)
        # A aptidão é maior quanto mais próximo o resultado estiver do alvo.
        # A aptidão máxima é 1.0 quando o resultado é igual ao alvo.
        return 1 / (1 + abs(self.target - result))

    def _roulette_wheel_selection(self):
        """Seleciona um pai usando o método de seleção da roleta."""
        total_fitness = sum(ind['fitness'] for ind in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in self.population:
            current += individual['fitness']
            if current > pick:
                return individual
        return self.population[-1]

    def _crossover(self, parent1, parent2):
        """Realiza um crossover de ponto único entre dois pais."""
        point = random.randint(1, len(self.search_ranges) - 1)
        keys = list(parent1.keys())
        child = {}
        for i, key in enumerate(keys):
            if i < point:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, chromosome):
        """Muta um cromossomo alterando aleatoriamente seus genes."""
        for key in chromosome:
            if random.random() < self.mutation_rate:
                chromosome[key] = random.randint(*self.search_ranges[key])
        return chromosome

    def run(self):
        """Executa o processo de evolução do algoritmo genético."""
        # 1. Inicializa a população
        self.population = [{'genes': self._create_chromosome()} for _ in range(self.population_size)]
        for individual in self.population:
            individual['fitness'] = self._calculate_fitness(individual['genes'])

        # 2. Loop de evolução
        for generation in range(self.num_generations):
            # Armazena o melhor indivíduo da geração atual
            best_individual = max(self.population, key=lambda ind: ind['fitness'])
            self.best_individuals_history.append((generation, best_individual))

            # Para se uma solução ótima for encontrada
            if best_individual['fitness'] == 1.0:
                break

            # Cria a próxima geração
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self._roulette_wheel_selection()
                parent2 = self._roulette_wheel_selection()
                child_genes = self._crossover(parent1['genes'], parent2['genes'])
                child_genes = self._mutate(child_genes)
                child_fitness = self._calculate_fitness(child_genes)
                new_population.append({'genes': child_genes, 'fitness': child_fitness})
            
            self.population = new_population

    def print_results(self):
        """Imprime os melhores indivíduos encontrados em cada geração."""
        print("\nMelhores indivíduos por geração:")
        for generation, individual in self.best_individuals_history:
            genes = individual['genes']
            fitness = individual['fitness']
            result = 5 * genes['x'] + genes['y']**2 + genes['w'] + genes['z']**3
            print(f"Geração {generation}: {genes} -> Resultado: {result}, Fitness: {fitness:.4f}")

# --- Execução Principal ---
if __name__ == "__main__":
    # Cria e executa o algoritmo
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        num_generations=NUM_GENERATIONS,
        search_ranges=SEARCH_RANGES,
        target=TARGET_RESULT
    )
    ga.run()

    # Exibe os resultados finais
    ga.print_results()
