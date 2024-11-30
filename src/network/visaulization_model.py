from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(n):
        if n not in nodes:
            nodes.add(n)
            for child in n._prev:
                edges.add((child, n))
                build(child)
    build(root)
    return nodes, edges


def draw_graph(root):
    dot = Digraph('Value Graph', engine='dot', format='png', node_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label=f'data: {n.data:.2f}', shape='record')
        if n._op:
            dot.node(name=str(id(n))+n._op, label=n._op, shape='circle')
            dot.edge(str(id(n))+n._op, str(id(n)))

    for e1, e2 in edges:
        dot.edge(str(id(e1)), str(id(e2))+e2._op)

    return dot

