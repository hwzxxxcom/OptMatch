import torch
import torch_geometric
import torch_geometric.nn

from gnns import GIN, Swish

class Estimator(torch.nn.Module):
    def __init__(self, input_query_size, input_graph_size, model_size, nlayer, **args):
        super().__init__()
        self.inq = torch.nn.Linear(input_query_size, model_size)
        self.ing = torch.nn.Linear(input_graph_size, model_size)
        self.qnn = GIN(model_size, model_size, nlayer)
        self.lin = torch.nn.Linear(model_size * 2, model_size)
        self.res = torch.nn.Linear(model_size * 2, model_size)
        self.act = torch.nn.Linear(model_size * 2, 4) # 0 1 2 3
        self.out = torch.nn.Linear(model_size, 2)
        self.dis_est = torch.nn.Linear(model_size, model_size)
        self.dis_can = torch.nn.Linear(model_size, model_size)
        self.dis_est2 = torch.nn.Linear(model_size, model_size)
        self.dis_can2 = torch.nn.Linear(model_size, model_size)
        self.args = args
    def estimate_counts(self, query_x, query_e, graph_x, matches):
        # est_trk, est_cnt
        device = next(self.parameters()).device
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64).to(device)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.lin(inter_x)
        return torch.relu(self.out(torch.max(inter_x, dim = 0).values))
    def estimate_next(self, query_x, query_e, graph_x, matches):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64, requires_grad=False)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.lin(inter_x)
        return inter_x
    def estimate_bad_match(self, query_x, query_e, graph_x, matches):
        # No gradient
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        counts = []
        for i in range(matches.shape[1]):
            partial_matches = matches @ (torch.eye(matches.shape[1], dtype=torch.int64)[:, [ii for ii in range(matches.shape[1]) if ii != i]])
            idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
            for (u, v) in partial_matches.T:
                idx[u] = v
            graph_xx = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
            inter_x = torch.cat([query_x, graph_xx], dim = -1)
            inter_x = torch.relu(self.lin(inter_x))
            counts.append(torch.relu(self.out(torch.max(inter_x, dim = 0).values)))
        counts = torch.stack(counts)
        match_rate = counts[:, 1] / counts[:, 0]
        bad_match = matches[0, match_rate.argmax()]
        return bad_match
    def estimate_good_match_res(self, query_x, query_e, graph_x, matches):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.res(inter_x)
        return inter_x
    def estimate_action(self, query_x, query_e, graph_x, matches):
        # 0 1 2 3
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.act(inter_x)
        return torch.softmax(self.out(torch.max(inter_x, dim = 0).values), dim = -1)
    def evaluate_next(self, next_h, candidate_h):
        h_a, h_b = self.dis_est2(torch.relu(self.dis_est(next_h))), self.dis_can2(torch.relu(self.dis_can(candidate_h)))
        return torch.sqrt(torch.sum((h_a - h_b) ** 2, dim = -1))

class Estimator(torch.nn.Module):
    def __init__(self, input_query_size, input_graph_size, model_size, nlayer, **args):
        super().__init__()
        self.model_size = model_size
        self.inq = torch.nn.Linear(input_query_size, model_size)
        self.ing = torch.nn.Linear(input_graph_size, model_size)
        self.qnn = GIN(model_size, model_size, nlayer)
        self.lin = torch.nn.Linear(model_size * 2, model_size)
        self.res = torch.nn.Linear(model_size * 2, model_size)
        self.act = torch.nn.Linear(model_size * 2, 4) # 0 1 2 3
        self.out = torch.nn.Linear(model_size, 2)
        self.dis_est = torch.nn.Linear(model_size, model_size)
        self.dis_can = torch.nn.Linear(model_size, model_size)
        self.dis_est2 = torch.nn.Linear(model_size, model_size)
        self.dis_can2 = torch.nn.Linear(model_size, model_size)
        self.args = args
    def estimate_counts(self, query_x, query_e, graph_x, matches):
        # est_trk, est_cnt
        device = next(self.parameters()).device
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64).to(device)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        #graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        graph_x = torch.zeros_like(query_x) + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.lin(inter_x)
        return torch.relu(self.out(torch.max(inter_x, dim = 0).values))
    def estimate_next(self, query_x, query_e, graph_x, matches):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64, requires_grad=False)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.lin(inter_x)
        return inter_x
    def estimate_bad_match(self, query_x, query_e, graph_x, matches):
        # No gradient
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        counts = []
        for i in range(matches.shape[1]):
            partial_matches = matches @ (torch.eye(matches.shape[1], dtype=torch.int64)[:, [ii for ii in range(matches.shape[1]) if ii != i]])
            idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
            for (u, v) in partial_matches.T:
                idx[u] = v
            graph_xx = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
            inter_x = torch.cat([query_x, graph_xx], dim = -1)
            inter_x = torch.relu(self.lin(inter_x))
            counts.append(torch.relu(self.out(torch.max(inter_x, dim = 0).values)))
        counts = torch.stack(counts)
        match_rate = counts[:, 1] / counts[:, 0]
        bad_match = matches[0, match_rate.argmax()]
        return bad_match
    def estimate_good_match_res(self, query_x, query_e, graph_x, matches):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.res(inter_x)
        return inter_x
    def estimate_action(self, query_x, query_e, graph_x, matches):
        # 0 1 2 3
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.act(inter_x)
        return torch.softmax(self.out(torch.max(inter_x, dim = 0).values), dim = -1)
    def evaluate_next(self, next_h, candidate_h):
        h_a, h_b = self.dis_est2(torch.relu(self.dis_est(next_h))), self.dis_can2(torch.relu(self.dis_can(candidate_h)))
        #return torch.sqrt(torch.sum((h_a - h_b) ** 2, dim = -1))
        return torch.sqrt(torch.sum((next_h - candidate_h) ** 2, dim = -1))
    
class OptMatchEstimator(Estimator):
    def __init__(self, input_query_size, input_graph_size, model_size, nlayer, **args):
        super().__init__(input_query_size, input_graph_size, model_size, nlayer, **args)
        self.WQ = torch.nn.Linear(model_size, model_size)
        self.WK = torch.nn.Linear(model_size, model_size)
        self.WV = torch.nn.Linear(model_size, model_size)
        self.qnn1 = GIN(model_size * 2, model_size, nlayer)
        self.matchest = torch.nn.Linear(model_size, 1)
        
    def estimate_counts_(self, query_x, query_e, graph_x, matches):
        # est_trk, est_cnt
        device = next(self.parameters()).device
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64).to(device)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        #graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        graph_x = torch.zeros_like(query_x) + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.lin(inter_x)
        return torch.relu(self.out(torch.max(inter_x, dim = 0).values))
    def estimate_counts(self, query_x, query_e, graph_x, matches):
        # est_trk, est_cnt
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64, requires_grad=False)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        # graph_h = torch.zeros_like(query_x) + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]
        # graph_h = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        graph_h = (torch.zeros_like(query_x) 
                   + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx] )
        #           + (idx.reshape([-1, 1]) == torch.tensor(-1)) * torch.relu((self.WQ(query_x)@self.WK(graph_x[candidates]).T)) @ self.WV(graph_x[candidates]))
        inter_x = torch.cat([query_x, graph_h], dim = -1)
        query_h = self.qnn1(inter_x, query_e) 
        query_h = query_h / (query_h.norm(dim = -1).reshape([-1, 1]))
        return torch.relu(self.out(torch.max(query_h, dim = 0).values))
    def estimate_next_(self, query_x, query_e, graph_x, matches, candidates):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64, requires_grad=False)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        #graph_h = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        graph_h = torch.zeros_like(query_x) + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]
        inter_x = torch.cat([query_x, graph_h], dim = -1)
        inter_x = self.lin(inter_x)
        # import random
        # if random.random() < 0.0001: 
        #     print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1))
        #     print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1) @ self.WV(graph_x[candidates]))
        #print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1))
        return torch.sigmoid((self.WQ(inter_x)@self.WK(graph_x[candidates]).T)) @ self.WV(graph_x[candidates])
    def estimate_next(self, query_x, query_e, graph_x, matches, candidates):
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64, requires_grad=False)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        # graph_h = torch.zeros_like(query_x) + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]
        # graph_h = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        graph_h = (torch.zeros_like(query_x) 
                   + (idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx] 
                   + (idx.reshape([-1, 1]) == torch.tensor(-1)) * torch.relu((self.WQ(query_x)@self.WK(graph_x[candidates]).T)) @ self.WV(graph_x[candidates]))
        inter_x = torch.cat([query_x, graph_h], dim = -1)
        query_h = self.qnn1(inter_x, query_e) 
        query_h = query_h / (query_h.norm(dim = -1).reshape([-1, 1]))
        #inter_x = self.lin(inter_x)
        # import random
        # if random.random() < 0.0001: 
        #     print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1))
        #     print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1) @ self.WV(graph_x[candidates]))
        #print(torch.log_softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T), dim = -1))
        return query_h
    def estimate_bad_match(self, query_x, query_e, graph_x, matches):
        # No gradient
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        counts = []
        for i in range(matches.shape[1]):
            partial_matches = matches @ (torch.eye(matches.shape[1], dtype=torch.int64)[:, [ii for ii in range(matches.shape[1]) if ii != i]])
            idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
            for (u, v) in partial_matches.T:
                idx[u] = v
            graph_xx = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
            inter_x = torch.cat([query_x, graph_xx], dim = -1)
            inter_x = torch.relu(self.lin(inter_x))
            counts.append(torch.relu(self.out(torch.max(inter_x, dim = 0).values)))
        counts = torch.stack(counts)
        match_rate = counts[:, 1] / counts[:, 0]
        bad_match = matches[0, match_rate.argmax()]
        return bad_match
    # def estimate_good_match_res(self, query_x, query_e, graph_x, matches, candidates):
    #     idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
    #     for (u, v) in matches.T:
    #         idx[u] = v
    #     query_x = self.inq(query_x)
    #     query_x = self.qnn(query_x, query_e)
    #     graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
    #     inter_x = torch.cat([query_x, graph_x], dim = -1)
    #     inter_x = self.res(inter_x)
    #     inter_x = inter_x + query_x
    #     return (torch.softmax((self.WQ(inter_x)@self.WK(graph_x[candidates]).T) / torch.tensor(self.model_size), dim = -1) @ graph_x[candidates]).shape
    def estimate_action(self, query_x, query_e, graph_x, matches):
        # 0 1 2 3
        idx = -torch.ones([query_x.shape[0]], dtype=torch.int64)
        for (u, v) in matches.T:
            idx[u] = v
        query_x = self.inq(query_x)
        query_x = self.qnn(query_x, query_e)
        graph_x = torch.zeros_like(query_x) + torch.relu(self.ing((idx.reshape([-1, 1]) != torch.tensor(-1)) * graph_x[idx]))
        inter_x = torch.cat([query_x, graph_x], dim = -1)
        inter_x = self.act(inter_x)
        return torch.softmax(self.out(torch.max(inter_x, dim = 0).values), dim = -1)
    def evaluate_next(self, next_h, candidate_h):
        #h_a, h_b = self.dis_est2(torch.relu(self.dis_est(next_h))), self.dis_can2(torch.relu(self.dis_can(candidate_h)))
        return torch.sqrt(torch.sum((next_h - candidate_h) ** 2, dim = -1))

def evaluate_bunch(counter, matcher:Estimator, query_x, query_e, graph_x, matches, order, 
                   candidates, level = 2, width = 2):
    candidates = candidates[1]
    level = min(len(order) - matches.shape[1], level)
    count = 0
    sum_rate = torch.tensor(0., device=query_x.device)
    rates = []
    matched = set(matches[0].tolist())
    tobechecked = [ord for ord in order if ord not in matched]
    cur_candidates = candidates[tobechecked[0]]
    next_h = matcher.estimate_next(query_x, query_e, graph_x, matches, cur_candidates)[tobechecked[0]]
    candidate_emb = graph_x[cur_candidates]
    dif = (matcher.dis_est2(torch.relu(matcher.dis_est(next_h))) - 
           matcher.dis_can2(torch.relu(matcher.dis_can(candidate_emb))))
    cur_selects = [cur_candidates[i] for i in torch.sum(dif * dif, dim = -1).topk(min(width, len(cur_candidates)), largest=False).indices]
    queue = [([cs], 1) for cs in cur_selects]
    while queue:
        cur_selected, depth = queue[0]
        del queue[0] 
        est_trk, est_cnt = counter.estimate_counts(query_x, query_e, graph_x, torch.cat([matches, torch.tensor([tobechecked[:len(cur_selected)], cur_selected])], dim = -1))
        #print(est_trk, est_cnt)
        rate = est_cnt / est_trk if est_trk != 0 else torch.tensor(0.)
        rates.append(rate)
        sum_rate += rate
        count += 1
        if depth < level:
            cur_candidates = candidates[int(tobechecked[depth])]
            next_h = matcher.estimate_next(query_x, query_e, graph_x, torch.cat([matches, torch.tensor([tobechecked[:len(cur_selected)], cur_selected])], dim = -1), cur_candidates)[tobechecked[depth]]
            candidate_emb = graph_x[cur_candidates]
            dif = (matcher.dis_est2(torch.relu(matcher.dis_est(next_h))) - 
                   matcher.dis_can2(torch.relu(matcher.dis_can(candidate_emb))))
            cur_selects = [cur_candidates[i] for i in torch.sum(dif * dif, dim = -1).topk(min(width, len(cur_candidates)), largest=False).indices]
            for cs in cur_selects:
                queue.append((cur_selected + [cs], 2))
    return torch.tensor(0) if count == 0 else max(rates)
    # return torch.tensor(0) if count == 0 else sum_rate / count

if __name__ == '__main__':

    query_x, query_e, graph_x, matches = (torch.ones([5, 71]).to('cuda'), torch.tensor([[0, 1, 1, 2, 3], [1, 2, 3, 3, 4]]).to('cuda'), torch.ones([10000, 64]).to('cuda'), torch.tensor([[0,1,2,3,4],[5,7,3,9,10]]).to('cuda'))
    query_x, query_e, graph_x, matches = (torch.ones([5, 71]), torch.tensor([[0, 1, 1, 2, 3], [1, 2, 3, 3, 4]]), torch.ones([10000, 64]), torch.tensor([[0,3,4],[5,38,42]]))
    candidates = [[0,1,2,3,4,5,6,7,8,9,10], 
                  list(range(10, 20)), 
                  list(range(20, 30)), 
                  list(range(30, 40)),
                  list(range(40, 50))]
    order = [0,2,3,1,4]

    estimator = Estimator(71, 64, 64, 2)

    import time
    def f():
        s = time.time()
        for i in range(100000): _ = estimator.estimate_counts(query_x, query_e, graph_x, matches)
        print((time.time() - s) / 100000)
        