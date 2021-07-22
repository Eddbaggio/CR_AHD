import logging
import random
from abc import ABC, abstractmethod
from typing import Iterable, Sequence, List, Callable, Union

import numpy as np
from sklearn.cluster import KMeans
import tqdm

from src.cr_ahd.auction_module import bundle_valuation as bv
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)

"""
auction_request_pool: Sequence[int][int]
    a sequence of request indices of the requests that were submitted to be auctioned
    
auction_bundle_pool: Sequence[Sequence[int]]
    a nested sequence of request index sequences. Each inner sequence is a {bundle} inside the auction_bundle_pool that
    carriers will have to bid on
    
bundle: Sequence[int]
    a sequence of request indices that make up one bundle -> cannot have duplicates etc., maybe a set rather than a list
    would be better?
    
bundling: Sequence[Sequence[int]]
    a sequence of {bundles} (see above) that fully partition the {auction_request_pool}
    
bundling_labels: Sequence[int]
    a sequence of bundle indices that partitions the {auction_request_pool}
     NOTE: Contrary to the {bundling}, the {bundling_labels} is not nested and does not contain request indices but 
     bundle indices
"""


class BundlePoolGenerationBehavior(ABC):
    def __init__(self, num_auction_bundles: int, **kwargs):
        self.num_auction_bundles = num_auction_bundles
        self.parameters = kwargs

    def execute_bundle_pool_generation(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                       auction_request_pool: Sequence[int],
                                       original_bundling_labels: Sequence[int]):
        # random.seed(0)
        auction_bundle_pool = self._generate_auction_bundles(instance, solution, auction_request_pool,
                                                             original_bundling_labels)
        return auction_bundle_pool

    @abstractmethod
    def _generate_auction_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        pass


class AllBundles(BundlePoolGenerationBehavior):
    """
    creates the power set of all the submitted requests, i.e. all subsets of size k for all k = 1, ..., len(pool).
    Does not include emtpy set. NOTE: Ignores the num_auction_bundles parameter
    """

    def _generate_auction_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        return tuple(ut.power_set(range(len(auction_request_pool)), False))


class RandomMaxKPartition(BundlePoolGenerationBehavior):
    """
    creates self.num_auction_bundles random partitions of the submitted bundles with AT MOST as many subsets as there
    are carriers
    """

    def _generate_auction_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        bundles = []
        num_bundles = 0
        while num_bundles < self.num_auction_bundles - max(original_bundling_labels) + 1:
            bundle = self._random_max_k_partition_idx(auction_request_pool, max_k=instance.num_carriers)
            if bundle not in bundles:
                bundles.append(bundle)
                num_bundles += max(bundle) + 1
        bundles.append(original_bundling_labels)
        return bundles

    @staticmethod
    def _random_max_k_partition_idx(ls: Sequence, max_k: int) -> List[int]:
        """create a random paritioning of ls of at most max_k bins. returns a list of bin indices that map to ls"""
        if max_k < 1:
            return []
        # randomly determine the actual k
        k = random.randint(1, min(max_k, len(ls)))
        # We require that this list contains k different values, so we start by adding each possible different value.
        indices = list(range(k))
        # now we add random values from range(k) to indices to fill it up to the length of ls
        indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
        # shuffle the indices into a random order
        random.shuffle(indices)
        return indices


class SingleKMeansBundle(BundlePoolGenerationBehavior):
    """
    creates a k-means partitions of the submitted requests. generates exactly as many clusters as there are carriers.
    NOTE: ignores the num_auction_bundles
    :return
    """

    def _generate_auction_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        request_midpoints = [ut.midpoint(instance, *instance.pickup_delivery_pair(sr)) for sr in auction_request_pool]
        # k_means = KMeans(n_clusters=instance.num_carriers, random_state=0).fit(request_midpoints)
        k_means = KMeans(n_clusters=instance.num_carriers).fit(request_midpoints)
        return k_means.labels_


class GeneticAlgorithm(BundlePoolGenerationBehavior):
    @pr.timing
    def _generate_auction_bundles(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                  auction_request_pool: Sequence[int],
                                  original_bundling_labels: Sequence[int]):
        # parameters
        population_size: int = self.parameters['population_size']
        num_generations: int = self.parameters['num_generations']
        mutation_rate: float = self.parameters['mutation_rate']
        # only a fraction of generation_gap is replaced in a new gen. the remaining individuals (generation overlap)
        # are the top (1-generation_gap)*100 % from the previous gen, measured by their fitness
        generation_gap: float = self.parameters['generation_gap']
        bundle_valuation: bv.BundleValuation = self.parameters['bundle_valuation']

        fitness, population = self.initialize_population(instance,
                                                         solution,
                                                         auction_request_pool,
                                                         original_bundling_labels,
                                                         instance.num_carriers,
                                                         population_size,
                                                         bundle_valuation)

        for generation_counter in tqdm.trange(1, num_generations, desc='Bundle Generation', disable=False):

            # initialize new generation with the elites from the previous generation
            elites = ut.argsmax(fitness, int(population_size * (1 - generation_gap)))
            new_population: List[Sequence[int]] = [population[i] for i in elites]
            new_fitness: List[float] = [fitness[i] for i in elites]

            offspring_counter = 0
            while offspring_counter < population_size * generation_gap:

                # parent selection (get the parent's index first, then the actual parent string/chromosome)
                parent1, parent2 = self._roulette_wheel(fitness, 2)
                parent1 = population[parent1]
                parent2 = population[parent2]

                offspring = self.generate_offspring(instance, solution, auction_request_pool, parent1, parent2,
                                                    mutation_rate)

                # check for duplicates
                if offspring in new_population:
                    continue

                else:
                    # fitness evaluation
                    offspring_fitness = bundle_valuation.evaluate_bundle(instance, solution, offspring,
                                                                         auction_request_pool)

                    # add offspring to the new gen and increase population size counter
                    new_population.append(offspring)
                    new_fitness.append(offspring_fitness)
                    offspring_counter += 1

            # replace the old generation with the new one
            population = new_population
            fitness = new_fitness
            generation_counter += 1

        # select the best bundles
        limited_bundle_pool = self.generate_bundle_pool(auction_request_pool, fitness, population,
                                                        self.num_auction_bundles)

        # add each of the original bundles if it is not contained yet - this cannot be infeasible
        # this might exceed the auction_pool_size even more than self.generate_bundle_pool
        self._normalize_individual(original_bundling_labels)
        original_bundling_labels = [
            np.array(auction_request_pool)[np.array(original_bundling_labels) == bundle_idx].tolist()
            for bundle_idx in range(max(original_bundling_labels) + 1)]
        for ob in original_bundling_labels:
            if ob not in limited_bundle_pool:
                limited_bundle_pool.append(ob)

        return limited_bundle_pool

    @staticmethod
    def generate_bundle_pool(auction_pool, fitness, population: Sequence[Sequence[int]], pool_size):
        """
        create the set of bundles that is offered in the auction (carrier must solve routing to place bids on these)
        pool_size may be exceeded to guarantee that ALL bundles of a candidate solution are in the pool (either all or
        none can be in the solution).
        """
        auction_pool_array = np.array(auction_pool)
        bundle_pool = []

        # select the top n candidates, where n = ut.NUM_BUNDLES_TO_AUCTION
        population_sorted = (bundle for fit, bundle in sorted(zip(fitness, population), reverse=True))
        while len(bundle_pool) < pool_size:
            candidate_solution = next(population_sorted)
            candidate_solution = np.array(candidate_solution)
            for bundle_idx in range(max(candidate_solution) + 1):
                bundle = auction_pool_array[candidate_solution == bundle_idx].tolist()
                if bundle not in bundle_pool:
                    bundle_pool.append(bundle)

        return bundle_pool

    def generate_offspring(self, instance, solution, auction_request_pool, parent1, parent2, mutation_rate):
        """
        :param instance:
        :param solution:
        :param auction_request_pool:
        :param parent1:
        :param parent2:
        :param mutation_rate:
        :return: the NORMALIZED offspring
        """
        # crossover
        crossover_func: Callable = random.choice([self._crossover_uniform, self._crossover_geo])
        offspring: List[int] = crossover_func(instance, solution, auction_request_pool, parent1, parent2)
        # normalization IN PLACE
        self._normalize_individual(offspring)

        # mutation
        if random.random() <= mutation_rate:
            mutation_func: Callable = random.choice(
                [
                    self._mutation_move,
                    self._mutation_create,
                    self._mutation_join,
                    self._mutation_shift
                ])
            mutation_func(instance, solution, offspring, auction_request_pool)
        # normalization IN PLACE
        self._normalize_individual(offspring)

        return offspring

    def initialize_population(self, instance: it.PDPInstance, solution: slt.CAHDSolution, auction_pool: Sequence,
                              original_bundles: Sequence, n: int, population_size: int,
                              bundle_fitness: bv.BundleValuation):
        """
        initializes the a population of size population_size. this first generation includes the original bundles as
        well as a k-means bundling.

        :param instance:
        :param solution:
        :param auction_pool:
        :param original_bundles:
        :param n:
        :param population_size:
        :return: fitness and population
        """
        population = []
        fitness = []

        # initialize at least one k-means bundle that is also likely to be feasible
        k_means_individual = list(
            SingleKMeansBundle(self.num_auction_bundles).execute_bundle_pool_generation(instance, solution,
                                                                                        auction_pool, None))
        self._normalize_individual(k_means_individual)
        if k_means_individual not in population:
            population.append(k_means_individual)
            fitness.append(bundle_fitness.evaluate_bundle(instance, solution, k_means_individual, auction_pool))

        # fill the rest of the population with random individuals
        i = 1
        while i < population_size:
            individual = ut.random_max_k_partition_idx(auction_pool, n)
            self._normalize_individual(individual)
            if individual in population:
                continue
            else:
                population.append(individual)
                fitness.append(bundle_fitness.evaluate_bundle(instance, solution, individual, auction_pool))
                i += 1

        return fitness, population

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
    def _crossover_uniform(instance: it.PDPInstance, solution: slt.CAHDSolution, auction_request_pool, parent1,
                           parent2):
        """
        For each request, the corresponding bundle is randomly chosen from parent A or B. This corresponds to the
        uniform crossover of Michalewicz (1996), where only one child is produced.
        """
        offspring = []
        for i in range(len(parent1)):
            offspring.append(random.choice([parent1[i], parent2[i]]))
        return offspring

    @staticmethod
    def _crossover_geo(instance: it.PDPInstance, solution: slt.CAHDSolution, auction_request_pool, parent1, parent2):
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

        for i, request in enumerate(auction_request_pool):
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
    def _mutation_move(instance: it.PDPInstance, solution: slt.CAHDSolution, offspring: List[int],
                       auction_request_pool: Sequence[int]):
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
    def _mutation_create(instance: it.PDPInstance, solution: slt.CAHDSolution, offspring: List[int],
                         auction_request_pool: Sequence[int]):
        """
        A new bundle is created. We randomly chose one request and assign it to the new bundle. If by this the
        maximum number of bundles is exceeded, i.e., if there are more bundles than carriers (see Sect. 4),
        two randomly chosen bundles are merged.
        """
        new_bundle_idx = max(offspring) + 1
        # replace random position with new bundle
        offspring[random.randint(0, len(offspring) - 1)] = new_bundle_idx

        # merge two random bundles if the new exceeds the num_carriers
        if new_bundle_idx >= instance.num_carriers:
            rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, new_bundle_idx + 1), k=2)
            for i in range(len(offspring)):
                if offspring[i] == rnd_bundle_1:
                    offspring[i] = rnd_bundle_2
        pass

    @staticmethod
    def _mutation_join(instance: it.PDPInstance, solution: slt.CAHDSolution, offspring: List[int],
                       auction_request_pool: Sequence[int]):
        """
        Two randomly chosen bundles are merged. If the offspring has only a single bundle, nothing happens
        """
        num_bundles = max(offspring)
        if num_bundles >= 2:
            rnd_bundle_1, rnd_bundle_2 = random.sample(range(0, num_bundles + 1), k=2)
            for i in range(len(offspring)):
                if offspring[i] == rnd_bundle_1:
                    offspring[i] = rnd_bundle_2
        pass

    @staticmethod
    def _mutation_shift(instance: it.PDPInstance, solution: slt.CAHDSolution, offspring: List[int],
                        auction_request_pool: Sequence[int]):
        """
        for each of the given bundles in the candidate solution, the centroid is calculated. Then, requests are
        assigned to bundles according to their closeness to the bundle’s centroids.
        """
        bundles = ut.indices_to_nested_lists(offspring, auction_request_pool)
        centroids = []
        for bundle in bundles:
            direct_travel_dist = bv.bundle_direct_travel_dist(instance, bundle)
            centroid = bv.bundle_centroid(instance, bundle, direct_travel_dist)
            centroids.append(centroid)

        for i, request in enumerate(auction_request_pool):
            midpoint = ut.midpoint(instance, *instance.pickup_delivery_pair(request))

            min_distance = float('inf')
            closest_centroid = None
            for c, centroid in enumerate(centroids):
                distance = ut.euclidean_distance(*midpoint, *centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = c

            offspring[i] = closest_centroid

        pass


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
