import argparse
import random 
import time 
import torch
import datetime 

from estimator import Estimator, OptMatchEstimator, evaluate_bunch
from utils import *
from tqdm import tqdm

RECORD_TIME = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

def dist_func(x1, x2):
    assert len(x2.shape) == 2
    return 1 - x1 @ x2.T / (x1.norm(dim = -1) * x2.norm(dim = -1)) 
    return torch.sqrt(torch.sum((x1 - x2) ** 2))
def com_loss(model:Estimator, pred, target):
    #return dist_func(pred, model.dis_can(target))
    return 1 - torch.sigmoid(model.matchest(target - pred))
    return dist_func(model.dis_est2(torch.relu(model.dis_est(pred))), model.dis_can2(torch.relu(model.dis_can(target))))
    return dist_func(pred, target)

def compute_loss(model:Estimator, queryx, querye, gemb, dpth, mis_tree, order, matches):
    pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]))[[order[dpth]]]
    loss = torch.tensor(0., requires_grad=True)
    pointer = 0
    margin = torch.tensor(1.)
    if dpth == queryx.shape[0] - 1:
        for pointer in range(mis_tree):
            target = gemb[[matches[pointer][dpth]]]
            #loss = loss + dist_func(pred, target)
            loss = loss + com_loss(model, pred, target)
            if (negloss := margin - com_loss(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                loss = loss + negloss
    elif dpth == queryx.shape[0] - 2:
        for cnt in mis_tree:
            cnt = cnt ** 0.5
            target = gemb[matches[pointer][dpth]]
            #loss = loss + cnt * dist_func(pred, target)
            loss = loss + cnt * com_loss(model, pred, target)
            if (negloss := margin - com_loss(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                loss = loss + cnt * negloss
            loss = loss + compute_loss(model, queryx, querye, gemb, dpth + 1, cnt, order, matches[pointer: pointer + cnt])
            pointer += cnt
    else:
        for cnt, sub_tree in mis_tree:
            cnt = cnt ** 0.5
            target = gemb[matches[pointer][dpth]]
            #loss = loss + cnt * dist_func(pred, target)
            loss = loss + cnt * com_loss(model, pred, target)
            if (negloss := margin - com_loss(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                loss = loss + cnt * negloss
            loss = loss + compute_loss(model, queryx, querye, gemb, dpth + 1, sub_tree, order, matches[pointer: pointer + cnt])
            pointer += cnt
    return loss

def compute_loss_and_step(model:Estimator, queryx, querye, gemb, dpth, mis_tree, order, matches, 
                          losspack, loss_count_pack, loss_func, losses, optimizer, batch_size):
    #print(dpth)
    #print(losspack, loss_count_pack)
    pointer = 0
    margin = torch.tensor(1.)
    if dpth == queryx.shape[0] - 1:
        for pointer in range(mis_tree):
            if type(model) == OptMatchEstimator:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]), 
                                           torch.stack([matches[pointer][dpth], torch.randint(gemb.shape[0],(1,))[0]]))[[order[dpth]]]
            else:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]))[[order[dpth]]]
            target = gemb[matches[pointer][dpth]].reshape([-1, gemb.shape[-1]])
            loss = loss_func(model, pred, target)
            if (negloss := margin - loss_func(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                #losspack[0] = losspack[0] + negloss
                loss = loss + negloss
            loss.backward()
            losses[-1] = (losses[-1][0] + float(loss), losses[-1][1] + 1)
            loss_count_pack[0] += 1
            if loss_count_pack[0] >= batch_size:
                #losspack[0] = losspack[0] / batch_size * (loss_count_pack[0] / batch_size)
                # optimizer.zero_grad()
                # losspack[0].backward()
                # print(2, 'losspack[0].backward()')
                optimizer.step()
                optimizer.zero_grad()
                losses[-1] = losses[-1][0] / losses[-1][1]
                losses.append((0., 0))
                #losses.append(float(losspack[0]) / loss_count_pack[0])
                #del losspack[0] 
                #losspack.append(torch.tensor(0., requires_grad=True))
                loss_count_pack[0] = 0
    elif dpth == queryx.shape[0] - 2:
        for cnt in mis_tree:
            if type(model) == OptMatchEstimator:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]), 
                                           torch.stack([matches[pointer][dpth], torch.randint(gemb.shape[0],(1,))[0]]))[[order[dpth]]]
            else:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]))[[order[dpth]]]
            target = gemb[matches[pointer][dpth]].reshape([-1, gemb.shape[-1]])
            loss = cnt * loss_func(model, pred, target)
            if (negloss := margin - loss_func(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                #losspack[0] = losspack[0] + cnt * negloss
                loss = loss + cnt * negloss
            loss.backward()
            losses[-1] = (losses[-1][0] + float(loss), losses[-1][1] + cnt)
            loss_count_pack[0] += cnt
            if loss_count_pack[0] >= batch_size:
                #losspack[0] = losspack[0] / batch_size * (loss_count_pack[0] / batch_size)
                #optimizer.zero_grad()
                #losspack[0].backward()
                #print(3, 'losspack[0].backward()')
                optimizer.step()
                optimizer.zero_grad()
                losses[-1] = losses[-1][0] / losses[-1][1]
                losses.append((0., 0))
                #del losspack[0] 
                #losspack.append(torch.tensor(0., requires_grad=True))
                loss_count_pack[0] = 0
            compute_loss_and_step(model, queryx, querye, gemb, dpth + 1, cnt, order, matches[pointer: pointer + cnt],
                                  losspack, loss_count_pack, loss_func, losses, optimizer, batch_size)
            pointer += cnt
    elif dpth < queryx.shape[0] - 2:
        for cnt, sub_tree in mis_tree:
            if type(model) == OptMatchEstimator:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]), 
                                           torch.stack([matches[pointer][dpth], torch.randint(gemb.shape[0],(1,))[0]]))[[order[dpth]]]
            else:
                pred = model.estimate_next(queryx, querye, gemb, torch.stack([order[:dpth], matches[0][:dpth]]))[[order[dpth]]]
            target = gemb[matches[pointer][dpth]].reshape([-1, gemb.shape[-1]])
            #losspack[0] = losspack[0] + cnt * loss_func(model, pred, target)
            loss = cnt * loss_func(model, pred, target)
            if (negloss := margin - loss_func(model, pred, gemb[[random.randint(0, gemb.shape[0] - 1)]])) > 0:
                #losspack[0] = losspack[0] + cnt * negloss
                loss = loss + cnt * negloss
            loss.backward()
            losses[-1] = (losses[-1][0] + float(loss), losses[-1][1] + cnt)
            loss_count_pack[0] += cnt
            if loss_count_pack[0] >= batch_size:
                #losspack[0] = losspack[0] / batch_size * (loss_count_pack[0] / batch_size)
                #optimizer.zero_grad()
                #losspack[0].backward()
                #print(1, 'losspack[0].backward()')
                optimizer.step()
                optimizer.zero_grad()
                losses[-1] = losses[-1][0] / losses[-1][1]
                losses.append((0., 0))
                #losses.append(float(losspack[0]) / loss_count_pack[0])
                #del losspack[0] 
                #losspack.append(torch.tensor(0., requires_grad=True))
                loss_count_pack[0] = 0
            compute_loss_and_step(model, queryx, querye, gemb, dpth + 1, sub_tree, order, matches[pointer: pointer + cnt],
                                  losspack, loss_count_pack, loss_func, losses, optimizer, batch_size)
            pointer += cnt

def train_matcher(model: Estimator, train_queryx, train_querye, train_orders, train_matches, gemb,
                  test_queryx, test_querye, test_orders, test_matches, epoch, batch_size):
    model.train()
    device = next(model.parameters()).device
    para_save_name = '../saved_models/%s_%d_%s/matcher.pth' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME)
    if not os.path.exists('../saved_models/%s_%d_%s' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME)): os.mkdir('../saved_models/%s_%d_%s' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME))
    # mis_trees = [get_mis_tree(train_matches[i]) for i in range(len(train_matches))]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_loss = 1e9
    def simplify_matches(matches):
        num = min(10 * np.sqrt(len(matches)), len(matches))
        selected = sorted(np.random.choice(list(range(len(matches))), int(num), replace = False).tolist())
        return matches[selected]
    ecount = 0
    for e in range(epoch): 
        if e >= 0 and e % 5 == 0:
            shuffled_list = list(range(len(train_queryx)))
            random.shuffle(shuffled_list)
            train_queryx  = [train_queryx[i]  for i in shuffled_list]
            train_querye  = [train_querye[i]  for i in shuffled_list]
            train_matches = [train_matches[i] for i in shuffled_list]
            partial_matches = [simplify_matches(matches) for matches in train_matches]
            mis_trees   = [get_mis_tree(partial_matches[i]) for i in range(len(partial_matches))] #[mis_trees[i]   for i in shuffled_list]
            train_orders  = train_orders[shuffled_list]
        losspack = [torch.tensor(0., requires_grad=True).to(device)]
        loss_count_pack = [0]
        losses = []
        optimizer.zero_grad()
        for i in range(len(train_queryx)):
            # print(i)
            losses.append((0., 0))
            compute_loss_and_step(model, train_queryx[i], train_querye[i], gemb, 0, mis_trees[i], train_orders[i], partial_matches[i],
                                  losspack, loss_count_pack, com_loss, losses, optimizer, batch_size)
            if loss_count_pack[0] != 0:
                #losspack[0] = losspack[0] / batch_size * (loss_count_pack[0] / batch_size)
                #optimizer.zero_grad()
                #print('losspack[0].backward ()', 4)
                #losspack[0].backward()
                optimizer.step()
                optimizer.zero_grad()
                losses[-1] = losses[-1][0] / losses[-1][1]
                #losses.append(float(losspack[0].detach()) / loss_count_pack[0])
                #del losspack[0] 
                #losspack.append(torch.tensor(0., requires_grad=True))
                loss_count_pack[0] = 0
            else:
                del losses[-1]
                loss_count_pack[0] = 0
        #print(losses)
        sum_loss = np.sum(losses) / len(losses)
        
        print('epoch: %d, loss: %.4f' % (e, sum_loss))
        if sum_loss < best_loss:
            print('currently the best')
            torch.save(model.state_dict(), para_save_name)
            best_loss = sum_loss
            ecount = 0
        else: ecount += 1
        if ecount >= 5: return True


# def train_matcher_old(model: Estimator, train_queryx, train_querye, train_orders, train_matches, gemb,
#                   test_queryx, test_querye, test_orders, test_matches, epoch, batchsize):
#     model.train()
#     device = next(model.parameters()).device
#     para_save_name = '../saved_models/%s_%s/matcher.pth' % (args.dataname, RECORD_TIME)
#     if not os.path.exists('../saved_models/%s_%s' % (args.dataname, RECORD_TIME)): os.mkdir('../saved_models/%s_%s' % (args.dataname, RECORD_TIME))
#     mis_trees = [get_mis_tree(train_matches[i]) for i in range(len(train_matches))]
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     best_loss = 1e9
#     for e in range(epoch): 
#         if e > 0 and e % 1 == 0:
#             #print(1)
#             shuffled_list = list(range(len(train_queryx)))
#             random.shuffle(shuffled_list)
#             train_queryx  = [train_queryx[i]  for i in shuffled_list]
#             train_querye  = [train_querye[i]  for i in shuffled_list]
#             train_matches = [train_matches[i] for i in shuffled_list]
#             mis_trees   = [mis_trees[i]   for i in shuffled_list]
#             train_orders  = train_orders[shuffled_list]
#         #for batch in range(batchsize):
#             #if (batch + 1) * batchsize < len(train_queryx): batch_len = len(train_queryx) % batchsize
#             #else: batch_len = batchsize
#         losses = []
#         for i in range(len(train_queryx)):
#             loss:torch.Tensor = compute_loss(model, train_queryx[i], train_querye[i], gemb, 0, mis_trees[i], train_orders[i], train_matches[i])
#             #if i == 0: print(loss.detach())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.append(float(loss.detach()))
#         sum_loss = np.sum(losses)

#         if sum_loss < best_loss:
#             print('epoch: %d, loss: %.4f' % (e, sum_loss / len(losses)))
#             torch.save(model.state_dict(), para_save_name)
#             best_loss = sum_loss    

def train_counter(model:Estimator, train_queryx, train_querye, train_orders, 
                  train_searches, train_tracks, train_counts, gemb, epoch = 100, batch_size = 64):
    device = next(model.parameters()).device
    model.train()
    para_save_name = '../saved_models/%s_%d_%s/counter.pth' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME)
    if not os.path.exists('../saved_models/%s_%d_%s' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME)): os.mkdir('../saved_models/%s_%d_%s' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_func = torch.nn.MSELoss()
    best_loss = torch.tensor(1.e9).to(device)
    ecount = 0
    for e in range(epoch): 
        if e > 0 and e % 5 == 0:
            print('%d in %d epoches' % (e, epoch))
            shuffled_list = list(range(len(train_queryx)))
            random.shuffle(shuffled_list)
            train_queryx  = [train_queryx[i]  for i in shuffled_list]
            train_querye  = [train_querye[i]  for i in shuffled_list]
            train_searches= [train_searches[i] for i in shuffled_list]
            train_tracks  = [train_tracks[i] for i in shuffled_list]
            train_counts  = [train_counts[i] for i in shuffled_list]
            train_orders  = train_orders[shuffled_list]
        losses = []
        sum_est_trk = [0]
        sum_est_cnt = [0] 
        for i in range(len(train_queryx)):
            loss = torch.tensor(0., requires_grad=True).to(device)
            loss_count = 0
            qryx, qrye, ordr, trks, schs, cnts = (x[i] for x in 
                [train_queryx, train_querye, train_orders, train_tracks, train_searches, train_counts])
            #print(len(schs))
            for j in range(len(schs)):
                if random.random() > len(schs) ** 0.65 / len(schs) and cnts[j] == 0: continue
                if random.random() > len(schs) ** 0.9 / len(schs) and cnts[j] != 0: continue
                mtch = torch.stack([ordr[:len(schs[j])], torch.tensor(schs[j]).to(device)])
                est_trk, est_cnt = model.estimate_counts(qryx, qrye, gemb, mtch)
                sum_est_trk[-1] += est_trk.detach()
                sum_est_cnt[-1] += est_cnt.detach()
                
                trk, cnt = trks[j], cnts[j]
                loss = loss + loss_func(est_trk, trk)
                loss = loss + loss_func(est_cnt, cnt)
                loss_count += 1
                if loss_count >= batch_size:
                    loss = loss / batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()) * (loss_count / batch_size))
                    loss = torch.tensor(0., requires_grad=True).to(device)
                    sum_est_trk[-1] = sum_est_trk[-1] / loss_count
                    sum_est_cnt[-1] = sum_est_cnt[-1] / loss_count
                    sum_est_cnt.append(0)
                    sum_est_trk.append(0)
                    loss_count = 0
                if torch.isnan(loss):
                    print(est_trk, est_cnt, trk, cnt)
                    raise
            if loss_count > 0:
                loss = loss / batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()) * (loss_count / batch_size))
                sum_est_trk[-1] = sum_est_trk[-1] / loss_count
                sum_est_cnt[-1] = sum_est_cnt[-1] / loss_count
                loss_count = 0
        print(thisepochloss:=(sum(losses) / len(losses)))
        # print(sum_est_cnt, sum_est_trk)
        if e >= 2 and sum(sum_est_cnt) * sum(sum_est_trk) == 0: 
            print('bad')
            return False
        if thisepochloss < best_loss:
            print('^        Currently the best.')
            torch.save(model.state_dict(), para_save_name)
            best_loss = thisepochloss
            ecount = 0
        else: ecount += 1
        if ecount >= 5: return True
    return True

def train_good_res_selecter(model: Estimator, train_queryx, train_querye, train_orders, 
                            train_searches, train_tracks, train_counts, graphx, train_candidates, 
                            counter: Estimator, matcher: Estimator, shallow_depth, deep_depth,
                            epoch = 100, sample_rate = .1):
    model.train() 
    para_save_name = '../saved_models/%s_%d_%s/selector.pth' % (args.dataname, train_queryx[0].shape[0], RECORD_TIME)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = dist_func # f(pred, target)
    best_loss = 1e9
    for e in range(epoch): 
        pntlosses = []
        if e > 0 and e % 5 == 0:
            print('%d in %d epoches' % (e, epoch))
            shuffled_list = list(range(len(train_queryx)))
            random.shuffle(shuffled_list)
            train_queryx  = [train_queryx[i]  for i in shuffled_list]
            train_querye  = [train_querye[i]  for i in shuffled_list]
            train_orders  = train_orders[shuffled_list]
            train_searches= [train_searches[i] for i in shuffled_list]
            train_tracks  = [train_tracks[i] for i in shuffled_list]
            train_counts  = [train_counts[i] for i in shuffled_list]
            train_candidates = [train_candidates[i] for i in shuffled_list]
        for idx in range(len(train_queryx)):
            loss = torch.tensor(0., requires_grad=True)
            qryx, qrye, ordr, trks, schs, cnts, cdds = (x[idx] for x in 
                [train_queryx, train_querye, train_orders, train_tracks, train_searches, train_counts, train_candidates])
            np.random.shuffle(sample_lst := list(range(len(schs))))
            sample_lst = sample_lst[:max(1, int(len(schs) * sample_rate))]
            sample_schs = [schs[i] for i in sample_lst]
            sample_trks = trks[sample_lst]
            sample_cnts = cnts[sample_lst]
            for sch in sample_schs:
                if not shallow_depth <= len(sch) < deep_depth: continue 
                scores = []
                losses = []
                '''
                for i in range(len(sch)):
                    mch_ord = [int(ordr[ii]) for ii in range(len(sch)) if ii != i]
                    mch_sch = [int(sch[ii]) for ii in range(len(sch)) if ii != i]
                    mch = torch.tensor([mch_ord, mch_sch], dtype=torch.long)
                    scores.append(float(evaluate_bunch(counter, matcher, qryx, qrye, graphx, mch, ordr, candidates).detach()))
                    h_res = model.estimate_good_match_res(qryx, qrye, graphx, mch)[ordr[i]]
                    cands = np.random.choice(candidates[ordr[i]], 3, False)
                    best_cand = -1
                    best_score = -1e9
                    for cand in cands:
                        tmp_sch = torch.tensor(sch)
                        tmp_sch[i] = cand
                        tmp_mch = torch.stack([ordr[:len(sch)], tmp_sch], dim = 0)
                        if float(evaluate_bunch(counter, matcher, qryx, qrye, graphx, tmp_mch, ordr, candidates).detach()) > best_score:
                            best_cand = cand
                    #print(h_res, best_cand, sch[i])
                    losses.append(matcher.evaluate_next(graphx[sch[i]] + h_res, graphx[best_cand]))'''
                
                H_res = model.estimate_good_match_res(qryx, qrye, graphx, torch.tensor([ordr.tolist()[:len(sch)], sch]))
                #print( torch.tensor([ordr.tolist()[:len(sch)], sch]))
                for i in range(len(sch)):
                    mch_ord = [int(ordr[ii]) for ii in range(len(sch)) if ii != i]
                    mch_sch = [int(sch[ii]) for ii in range(len(sch)) if ii != i]
                    mch = torch.tensor([mch_ord, mch_sch], dtype=torch.long)
                    scores.append(float(evaluate_bunch(counter, matcher, qryx, qrye, graphx, mch, ordr, cdds).detach()))
                    h_res = H_res[ordr[i]]
                    try:
                        cands = np.random.choice(cdds[ordr[i]], len(cdds[ordr[i]]) if len(cdds[ordr[i]]) < 3 else 3, False)
                        best_cand = -1
                        best_score = -1e9
                        for cand in cands:
                            tmp_sch = torch.tensor(sch)
                            tmp_sch[i] = cand
                            tmp_mch = torch.stack([ordr[:len(sch)], tmp_sch], dim = 0)
                            if float(evaluate_bunch(counter, matcher, qryx, qrye, graphx, tmp_mch, ordr, cdds).detach()) > best_score:
                                best_cand = cand
                        #print(h_res, best_cand, sch[i])
                        losses.append(matcher.evaluate_next(graphx[sch[i]] + h_res, graphx[best_cand]))
                    except:
                        pass

                scores = torch.tensor(scores)
                #print('s', scores)
                weights = torch.softmax(1 / (scores + .01 * (1 if (tmp:=scores.max() - scores.min()) == 0 else tmp)), dim = -1).detach()
                #print("lw", losses, weights)
                #print('sum', sum([ls * wt for (ls, wt) in zip(losses, weights)]))
                loss = loss + sum([ls * wt for (ls, wt) in zip(losses, weights)])
            #print('?', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pntlosses.append(loss.detach())
        print(thisepochloss := sum(pntlosses) / len(pntlosses))
        if thisepochloss < best_loss:
            print('^        Currently the best.')
            torch.save(model.state_dict(), para_save_name)
            best_loss = thisepochloss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='../data', type=str)
    parser.add_argument('--dataname', default='yeast', type=str)
    parser.add_argument('--qsize', default=71, type=int)
    parser.add_argument('--gsize', default=64, type=int)
    parser.add_argument('--hsize', default=64, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--nlayer', default=1, type=int)
    parser.add_argument('--nnode', default=8, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--training-rate', default=.8, type=float)
    parser.add_argument('--shallow-depth', default=2, type=int)
    parser.add_argument('--deep-depth', default=5, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model', default='optmatch', type=str)
    args = parser.parse_args()

    emb, queries, matches, orders, searches, tracks, counts, candidates = load_data(args.prefix, args.dataname, args.nnode)
    emb = emb.to(args.device)
    queries = preprocess_queries(queries, torch.eye(args.qsize))
    queryx = [queries[i][0] for i in range(len(queries))]
    querye = [queries[i][1] for i in range(len(queries))]
    orders = torch.tensor(orders)
    matches = [torch.tensor(matches[i]) for i in range(len(matches))]
    lst = list(range(len(queries)))[:40]
    np.random.seed(0)
    np.random.shuffle(lst)

    train_items = lst[:int(len(lst) * args.training_rate)]
    test_items  = lst[int(len(lst) * args.training_rate):]

    train_queryx  = [queryx[i].to(args.device)  for i in train_items]
    test_queryx   = [queryx[i].to(args.device)  for i in  test_items]
    train_querye  = [querye[i].to(args.device)  for i in train_items]
    test_querye   = [querye[i].to(args.device)  for i in  test_items]
    train_matches = [matches[i].to(args.device) for i in train_items]
    test_matches  = [matches[i].to(args.device) for i in  test_items]
    train_orders  = orders[train_items].to(args.device)
    test_orders   = orders[test_items].to(args.device)
    train_searches = [searches[i] for i in train_items]
    test_searches  = [searches[i] for i in test_items]
    train_tracks  = [torch.log(torch.tensor(tracks[i], dtype = torch.float32) + 1).to(args.device) for i in train_items]
    test_tracks   = [torch.log(torch.tensor(tracks[i], dtype = torch.float32) + 1).to(args.device) for i in test_items]
    train_counts  = [torch.log(torch.tensor(counts[i], dtype = torch.float32) + 1).to(args.device) for i in train_items]
    test_counts   = [torch.log(torch.tensor(counts[i], dtype = torch.float32) + 1).to(args.device) for i in test_items]
    train_candidates = [candidates[i] for i in train_items]
    test_candidates = [candidates[i] for i in test_items]
    #raise
    Estmtor = OptMatchEstimator if args.model == 'optmatch' else Estimator

    selector = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer).to(args.device)
    counter = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer).to(args.device)
    matcher = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer).to(args.device)

    while not train_counter(counter, train_queryx, train_querye, train_orders, train_searches, train_tracks, train_counts, emb, args.epoch, args.batch_size): 
        counter = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer).to(args.device)
    
    train_matcher(matcher, train_queryx, train_querye, train_orders, train_matches, emb,
                 test_queryx, test_querye, test_orders, test_matches, args.epoch, args.batch_size)
    train_good_res_selecter(selector, train_queryx, train_querye, train_orders, train_searches, train_tracks, train_counts, 
                        emb, train_candidates, counter, matcher, args.shallow_depth, args.deep_depth, args.epoch)
    