if __name__ == '__main__':
    pass
    # TODO write pseudo codes for ALL the stuff that's happening

    # TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

    # TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
    #  Alternatively, store the assignment in the instance? Or have some kind of AssignmentManager class?! Another
    #  possibility would be to have a class hierarchy for nodes: base_node, tw_node, depot_node, customer_node,
    #  assigned_node, ... <- using inheritance and polymorphism

    # TODO which of the @properties should be converted to proper class attributes, i.e. without delaying their
    #  computation? the @property may slow down the code, BUT in many cases it's probably a more idiot-proof way
    #  because otherwise I'd have to update the attribute which can easily be forgotten

    # TODO re-integrate animated plots for
    #  (1) static/dynamic sequential cheapest insertion construction => DONE
    #  (2) dynamic construction
    #  (3) I1 insertion construction => DONE

    # TODO's with * are from 06/12/20 or later from when I tried to understand my own code

    # TODO create class hierarchy! E.g. vertex (base, tw_vertex, depot_vertex, assigned_vertex, ...) and instance(
    #  base, centralized_instance, ...)


