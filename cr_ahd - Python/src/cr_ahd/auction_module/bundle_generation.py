import logging
import random
from abc import ABC, abstractmethod
from typing import Iterable, Sequence, List, Callable

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.cr_ahd.auction_module import bundle_valuation as bv
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class BundleSetGenerationBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence, original_bundles):
        bundle_set = self._generate_bundle_set(instance, solution, auction_pool, original_bundles)
        solution.bundle_generation = self.__class__.__name__
        return bundle_set

    @abstractmethod
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence,
                             original_bundles):
        pass


class AllBundles(BundleSetGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set.
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence,
                             original_bundles):
        return tuple(ut.power_set(auction_pool, False))  # todo GH decoding of a bundle pool!


class RandomMaxKPartition(BundleSetGenerationBehavior):
    """
    creates a random partition of the submitted bundles with AT MOST as many subsets as there are carriers
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence,
                             original_bundles):
        bundles = ut.random_max_k_partition(auction_pool, max_k=instance.num_carriers)
        return bundles  # todo GH decoding of a bundle pool!


class KMeansBundles(BundleSetGenerationBehavior):
    """
    creates a k-means partitions of the submitted requests. generates exactly as many clusters as there are carriers.

    :return a List of lists, where each sublist contains the cluster members of a cluster
    """

    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Iterable,
                             original_bundles):
        request_midpoints = [ut.midpoint(instance, *instance.pickup_delivery_pair(sr)) for sr in auction_pool]
        # k_means = KMeans(n_clusters=instance.num_carriers, random_state=0).fit(request_midpoints)
        k_means = KMeans(n_clusters=instance.num_carriers).fit(request_midpoints)
        return k_means.labels_


class GeneticAlgorithm(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence,
                             original_bundles: List[int]):
        # parameters
        population_size = 100
        # n = len(auction_pool)  # will produce infeasible candidates for the WDP
        n = instance.num_carriers  # to ensure a feasible candidate solution to the WDP
        num_generations = 100
        mutation_rate = 0.5
        # only a fraction of generation_gap is replaced in a new gen. the remaining individuals (generation overlap)
        # are the top (1-generation_gap)*100 % from the previous gen, measured by their fitness
        generation_gap = 0.9

        # initial population
        population = []
        fitness = []

        # add the original bundling as the first candidate - this cannot be infeasible
        self._normalize_individual(original_bundles)
        population.append(original_bundles)
        fitness.append(bv.GHProxyValuation(instance, solution, original_bundles, auction_pool))

        # initialize at least one k-means bundle that is also likely to be feasible
        k_means_individual = list(KMeansBundles().execute(instance, solution, auction_pool, None))
        self._normalize_individual(k_means_individual)
        if k_means_individual not in population:
            population.append(k_means_individual)
            fitness.append(bv.GHProxyValuation(instance, solution, k_means_individual, auction_pool))

        while len(population) < population_size - 1:
            individual = ut.random_max_k_partition_idx(auction_pool, n)
            self._normalize_individual(individual)
            if individual in population:
                continue
            else:
                population.append(individual)
                fitness.append(bv.GHProxyValuation(instance, solution, individual, auction_pool))

        # genetic algorithm
        best = ut.argmax(fitness)
        # print(f'Best in gen 1:\n'
        #       f'Index: {best}\n'
        #       f'Fitness: {fitness[best]}\n'
        #       f'Chromosome: {population[best]}\n')

        for generation_counter in tqdm(range(1, num_generations)):

            # initialize new generation with the elites from the previous generation
            elites = ut.argsmax(fitness, int(population_size * (1 - generation_gap)))
            new_population: List[Sequence[int]] = [population[i] for i in elites]
            new_fitness: List[int] = [fitness[i] for i in elites]

            offspring_counter = 0
            while offspring_counter < population_size * generation_gap:

                # parent selection (get the parent's index first, then the actual parent string/chromosome)
                parent1, parent2 = self._roulette_wheel(fitness, 2)
                parent1 = population[parent1]
                parent2 = population[parent2]

                # crossover
                crossover_func: Callable = random.choice([self._crossover_uniform, self._crossover_geo])
                offspring: List[int] = crossover_func(instance, solution, auction_pool, parent1, parent2)

                # mutation
                if random.random() <= mutation_rate:
                    mutation_func: Callable = random.choice(
                        [
                            self._mutation_move,
                            self._mutation_create,
                            self._mutation_join,
                            # self._mutation_shift
                        ])
                    mutation_func(instance, solution, offspring)

                # normalization IN PLACE
                self._normalize_individual(offspring)

                # check for duplicates
                if offspring in new_population:  # TODO using a hash function this can probably be sped up
                    continue

                # fitness evaluation
                offspring_fitness = bv.GHProxyValuation(instance, solution, offspring, auction_pool)

                # add offspring to the new gen and increase population size counter
                new_population.append(offspring)
                new_fitness.append(offspring_fitness)
                offspring_counter += 1

            # replace the old generation with the new one
            population = new_population
            fitness = new_fitness
            generation_counter += 1

            best = ut.argmax(fitness)

        # print(f'Best in gen {generation_counter}:\n'
        #       f'Index: {best}\n'
        #       f'Fitness: {fitness[best]}\n'
        #       f'Chromosome: {population[best]}\n')

        # create the set of bundles that is offered in the auction (carrier must solve routing to place bids on these)
        auction_pool_array = np.array(auction_pool)
        bundle_pool = []
        population_sorted = (b for b, f in sorted(zip(population, fitness)))
        while len(bundle_pool) < ut.NUM_BUNDLES_TO_AUCTION:
            candidate_solution = next(population_sorted)
            candidate_solution = np.array(candidate_solution)
            for bundle_idx in range(max(candidate_solution)):
                bundle = auction_pool_array[candidate_solution == bundle_idx].tolist()
                if bundle not in bundle_pool:
                    bundle_pool.append(bundle)

        return bundle_pool

    @staticmethod
    def _normalize_individual(individual: List[int]):
        """creates a normalized representation of the individual IN PLACE"""
        mapping_idx = 0
        mapping: List[int] = [-1] * (
                len(individual) + 1)  # +1 because the CREATE mutation may exceed the valid num_bundles temporarily
        mapping[individual[0]] = mapping_idx
        individual[0] = mapping_idx

        for i in range(1, len(individual)):
            if mapping[individual[i]] == -1:
                mapping_idx += 1
                mapping[individual[i]] = mapping_idx
                individual[i] = mapping_idx
            else:
                individual[i] = mapping[individual[i]]
        pass

    @staticmethod
    def _roulette_wheel(fitness: Sequence[float], n: int = 2):
        fitness_adj = []
        for elem in fitness:
            if elem == -float('inf'):
                fitness_adj.append(0)
            else:
                fitness_adj.append(elem)
        parents = random.choices(range(len(fitness)), weights=fitness_adj, k=n)
        return parents

    @staticmethod
    def _crossover_uniform(instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool, parent1, parent2):
        """
        For each request, the corresponding bundle is randomly chosen from parent A or B. This corresponds to the
        uniform crossover of Michalewicz (1996), where only one child is produced.
        """
        offspring = []
        for i in range(len(parent1)):
            offspring.append(random.choice([parent1[i], parent2[i]]))
        return offspring

    @staticmethod
    def _crossover_geo(instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool, parent1, parent2):
        """
        In this operator, we try to keep potentially good parts of existing bundles by combining the parents using
        geographic information. First, we calculate the center of each request, which is the midpoint between pickup
        and delivery location. Then, we randomly generate two points (A and B) in the plane. If the center of a
        request is closer to A, it is assigned to the bundle given in parent A, but if it is closer to B, it gets the
        bundle given in parent B. If the new  solution consists of too many bundles, two randomly chosen bundles are
        merged.
        """
        # setup
        offspring = []
        min_x = min(instance.x_coords)
        max_x = max(instance.x_coords)
        min_y = min(instance.y_coords)
        max_y = max(instance.y_coords)

        for i, request in enumerate(auction_pool):
            pickup, delivery = instance.pickup_delivery_pair(request)

            # center of each request = midpoint between pickup and delivery
            request_center = ut.midpoint_(instance.x_coords[pickup], instance.y_coords[pickup],
                                          instance.x_coords[delivery], instance.y_coords[delivery])

            # two random points in the plane, a and b
            random_points_x = ut.linear_interpolation([random.random(), random.random()], min_x, max_x)
            random_points_y = ut.linear_interpolation([random.random(), random.random()], min_y, max_y)
            random_a = (random_points_x[0], random_points_y[0])
            random_b = (random_points_x[1], random_points_y[1])
            dist_a = ut.euclidean_distance(*request_center, *random_a)
            dist_b = ut.euclidean_distance(*request_center, *random_b)

            # copy bundle assignment based on the proximity to the nearest point
            if dist_a < dist_b:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])

        # merge two bundles if there are more bundles than allowed
        if max(offspring) >= instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, max(offspring) + 1), k=2)
            for i in range(len(offspring)):
                if offspring[i] == rnd_bundle_1:
                    offspring[i] = rnd_bundle_2
        return offspring

    @staticmethod
    def _mutation_move(instance: it.PDPInstance, solution: slt.GlobalSolution, offspring: List[int]):
        """
        A random number of randomly chosen positions is changed. However, the number of available bundles is not
        increased.
        """
        num_bundles = max(offspring)
        num_requests = len(offspring)

        for pos in random.sample(range(num_requests), k=random.randint(0, num_requests)):
            offspring[pos] = random.randint(0, num_bundles)
        pass

    @staticmethod
    def _mutation_create(instance: it.PDPInstance, solution: slt.GlobalSolution, offspring: List[int]):
        """
        A new bundle is created. We randomly chose one request and assign it to the new bundle. If by this the
        maximum number of bundles is exceeded, i.e., if there are more bundles than carriers (see Sect. 4),
        two randomly chosen bundles are merged.
        """
        new_bundle_idx = max(offspring) + 1
        # replace random position with new bundle
        offspring[random.randint(0, len(offspring) - 1)] = new_bundle_idx

        # merge two random bundles if the new exceeds the num_carriers
        if new_bundle_idx > instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, new_bundle_idx + 1), k=2)
            for i in range(len(offspring)):
                if offspring[i] == rnd_bundle_1:
                    offspring[i] = rnd_bundle_2
        pass

    @staticmethod
    def _mutation_join(instance: it.PDPInstance, solution: slt.GlobalSolution, offspring: List[int]):
        """
        Two randomly chosen bundles are merged.
        """
        rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, max(offspring) + 1), k=2)  # todo check the range
        for i in range(len(offspring)):
            if offspring[i] == rnd_bundle_1:
                offspring[i] = rnd_bundle_2
        pass

    @staticmethod
    def _mutation_shift(instance: it.PDPInstance, solution: slt.GlobalSolution, offspring: List[int]):
        """
        for each of the given bundles in the candidate solution, the centroid is calculated. Then, requests are
        assigned to bundles according to their closeness to the bundle’s centroids.
        """

        pass


class ProxyTest(BundleSetGenerationBehavior):
    def _generate_bundle_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, auction_pool: Sequence,
                             original_bundles):
        best_candidate_solution = None
        best_proxy_valuation = 0

        # candidate == auction pool
        candidate = [auction_pool]
        valuation = bv.GHProxyValuation(instance, solution, [auction_pool], auction_pool)
        if valuation > best_proxy_valuation:
            best_candidate_solution = candidate
            best_proxy_valuation = valuation

        # generate all partitions of size [2, 3, 4, ... instance.num_carriers] i.e. all possible candidate solutions
        for k in range(2, instance.num_carriers + 1):
            candidate_generator = algorithm_u(auction_pool, k)
            for candidate in tqdm(candidate_generator, total=stirling_second(len(auction_pool), instance.num_carriers)):
                valuation = bv.GHProxyValuation(instance, solution, candidate, auction_pool)

                if valuation > best_proxy_valuation:
                    best_candidate_solution = candidate
                    best_proxy_valuation = valuation

        return best_candidate_solution


def stirling_second(n, k):
    """
    Stirling numbers of the second kind: number of ways to partition a set of n objects into m non-empty subsets
    https://extr3metech.wordpress.com/2013/01/21/stirling-number-generation-using-python-code-examples/
    Stirling Algorithm
    Cod3d by EXTR3ME
    https://extr3metech.wordpress.com
    """
    n1 = n
    k1 = k
    if n <= 0:
        return 1

    elif k <= 0:
        return 0

    elif n == 0 and k == 0:
        return -1

    elif n != 0 and n == k:
        return 1

    elif n < k:
        return 0

    else:
        temp1 = stirling_second(n1 - 1, k1)
        temp1 = k1 * temp1
        return (k1 * (stirling_second(n1 - 1, k1))) + stirling_second(n1 - 1, k1 - 1)


def algorithm_u(ns, m):
    """
    https://codereview.stackexchange.com/a/1944

    Generates all set partitions with a given number of blocks. The total amount of k-partitions is given by the
     Stirling number of the second kind

    :param ns: sequence of integers to build the subsets from
    :param m: integer, smaller than ns, number of subsets
    :return:
    """
    assert m > 1

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


if __name__ == '__main__':
    pass
