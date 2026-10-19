from snare.graph import AccountGraph


def test_shared_device_joins_accounts():
    graph = AccountGraph()
    graph.link(1, device=100, address=-1)
    graph.link(2, device=100, address=-1)
    graph.link(3, device=200, address=-1)
    assert sorted(graph.component(1)) == [1, 2]
    assert graph.component(3) == [3]


def test_links_are_transitive_through_addresses():
    graph = AccountGraph()
    graph.link(1, device=100, address=500)
    graph.link(2, device=101, address=500)
    graph.link(3, device=101, address=501)
    assert sorted(graph.component(3)) == [1, 2, 3]


def test_preview_does_not_modify_the_graph():
    graph = AccountGraph()
    graph.link(1, device=100, address=-1)
    preview = graph.component_if_linked(9, device=100, address=-1)
    assert sorted(preview) == [1, 9]
    assert graph.component(1) == [1]


def test_subgraph_always_contains_the_account():
    graph = AccountGraph()
    for account in range(200):
        graph.link(account, device=7, address=-1)
    sub = graph.subgraph(150, limit=20)
    assert 150 in sub["accounts"]
    assert len(sub["accounts"]) == 20
