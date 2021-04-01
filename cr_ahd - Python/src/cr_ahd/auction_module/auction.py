from abc import ABC, abstractmethod
import src.cr_ahd.auction_module.request_selection as rs
import src.cr_ahd.auction_module.bundle_generation as bg
import src.cr_ahd.auction_module.bidding as bd
import src.cr_ahd.auction_module.winner_determination as wd
import src.cr_ahd.utility_module.utils
import src.cr_ahd.utility_module.utils as ut


class Auction(ABC):
    def execute(self, instance):
        self._run_auction(instance)
        instance.auction_mechanism = self
        pass

    @abstractmethod
    def _run_auction(self, instance):
        pass


class Auction_a(Auction):
    """
    Request Selection Behavior: Highest Insertion Cost Distance
    Bundle Generation Behavior: Random Partition
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance):
        submitted_requests = rs.HighestInsertionCostDistance().execute(instance,
                                                                       src.cr_ahd.utility_module.utils.NUM_REQUESTS_TO_SUBMIT)
        bundle_set = bg.RandomPartition(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)


class Auction_b(Auction):
    """
    Request Selection Behavior: Cluster (Gansterer & Hartl 2016)
    Bundle Generation Behavior: Random Partition
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance):
        submitted_requests = rs.Cluster().execute(instance, src.cr_ahd.utility_module.utils.NUM_REQUESTS_TO_SUBMIT)
        bundle_set = bg.RandomPartition(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)


class Auction_c(Auction):
    """
    Request Selection Behavior: Highest Insertion Cost Distance
    Bundle Generation Behavior: K-Means
    Bidding Behavior: I1 Travel Distance Increase
    Winner Determination Behavior: Lowest Bid
    """

    def _run_auction(self, instance):
        submitted_requests = rs.HighestInsertionCostDistance().execute(instance,
                                                                       src.cr_ahd.utility_module.utils.NUM_REQUESTS_TO_SUBMIT)
        bundle_set = bg.KMeansBundles(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)
