from .NetBuilder import Placeholders
from .TFParameters import BATCH_SIZE
from .Structures import Token, Nodes


def compute_rates(root_node: Token):
    if not root_node.is_leaf:
        len_children = len(root_node.children)
        for child in root_node.children:
            if len_children == 1:
                child.left_rate = .5
                child.right_rate = .5
            else:
                child.right_rate = child.pos / (len_children - 1.0)
                child.left_rate = 1.0 - child.right_rate
            compute_rates(child)


def prepare_sample(ast: Nodes, zero_index):
    pc = Placeholders()
    # pc.target = [r_index[ast.root_node.author]]
    compute_rates(ast.root_node)
    ast.non_leafs.sort(key=lambda x: x.index)
    ast.all_nodes.sort(key=lambda x: x.index)
    pc.root_nodes = [node.token_type for node in ast.non_leafs]
    pc.node_emb = [node.token_type for node in ast.all_nodes]
    pc.node_left_c = [node.left_rate for node in ast.all_nodes]
    pc.node_right_c = [node.right_rate for node in ast.all_nodes]
    zero_node_index = len(pc.node_emb)
    pc.node_emb.append(zero_index)
    pc.node_left_c.append(0.0)
    pc.node_right_c.append(0.0)
    max_children_len = max([len(node.children) for node in ast.non_leafs])

    def align_nodes(_nodes):
        result = [node.index for node in _nodes]
        while len(result) != max_children_len:
            result.append(zero_node_index)
        return result

    pc.node_children = [align_nodes(node.children) for node in ast.non_leafs]
    return pc


def generate_batches(data_set, emb_indexes, net, dropout):
    size = len(data_set) // BATCH_SIZE
    pc = net.placeholders
    batches = []
    for j in range(size):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        feed = {net.dropout: dropout}
        for i in range(BATCH_SIZE):
            feed.update(pc[i].assign(prepare_sample(d[i], emb_indexes)))
        batches.append(feed)
    return batches
