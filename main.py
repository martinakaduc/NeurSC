import torch
import torch.nn as nn
import torch_geometric
import warnings
import argparse
import os
import time
import math
from tqdm import tqdm
from utils import load_g_graph, load_p_data, load_baseline, generate_features, preprocess_data_edge, save_params, preprocess_query2data, load_iso_baseline
from filtering import Filtering
from preprocess import SampleSubgraph, train_and_test
from model import BasicCountNet, QErrorLoss, QErrorLikeLoss, AttentiveCountNet, WasserstainDiscriminator


warnings.filterwarnings('ignore')
encoding = 'utf-8'
parser = argparse.ArgumentParser()

parser.add_argument('--in_feat', type=int, default=64,
                    help='input feature dim')
parser.add_argument('--out_dim', type=int, default=64,
                    help='dimension of output representation')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dimension of hidden feature.')
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for training')
parser.add_argument('--pooling_method', type=str, default='sumpool',
                    help='pooling method used.')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--dis_iter_num', type=int, default=1,
                    help='how many iteration to train discriminator.')
parser.add_argument('--train_percent', type=float, default=0.8,
                    help='proportion of training instance')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='the balance coeeficient for loss function')
parser.add_argument('--train_method', type=str, default='Normal',
                    help='train method of model, can be Normal or Curriculum')
parser.add_argument('--sample_method', type=str, default='induced',
                    help='how to sample the subgraphs, can be induced or start')                  
parser.add_argument('--share_net', type=bool, default=False,
                    help='whether share parameters between two networks')
parser.add_argument('--coarsen_data', type=bool, default=False,
                    help='whether coarsen the sampled data graph.')
parser.add_argument('--graph_file', type=str, default='eu2005',
                    help='the graph file name')
parser.add_argument('--file_folder', type=str, default='../../subgraph_matching/data/dataset/eu2005/grf_data_graph/',
                    help='folder that contains the graph info')
parser.add_argument('--baseline_path', type=str, default='../baseline/',
                    help='folder that contains the baseline file')
parser.add_argument('--query_path', type=str, default='../../subgraph_matching/data/dataset/eu2005/grf_query_graph/',
                    help='folder that contains the baseline file')
parser.add_argument('--query_vertex_num', type=str, default='4',
                    help='number of query vertices')
parser.add_argument('--correspondence', type=bool, default='False',
                    help='how to get correspondence pair')
parser.add_argument('--model_name', type=str, default='basic',
                    help='name of training model')
parser.add_argument('--baseline_name', type=str, default='_baseline.txt',
                    help='suffix of baseline file')
parser.add_argument('--device', type=str, default='cpu',
                    help='the device used for training')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
print(args)
model_save_path = '../saved_models/'
result_save_path = '../saved_results/'
params_save_path = '../saved_params/'
current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
print(current_time)
save_name = args.graph_file+ '_' + args.model_name+'_' + args.query_vertex_num +'_'+ args.pooling_method +'_' + current_time
model_save_name = save_name + '.pt'
result_save_name = save_name + '.txt'
params_save_name = save_name + '.txt'
save_params(params_save_path+params_save_name, args)
print(torch.version.cuda)
if not torch.cuda.is_available():
    args.device = 'cpu'
with open(result_save_path+result_save_name, 'w') as f_result:
    f_result.write('file_name q_error predicted_cardinality true_cardinality filtering_time total_time\n')


def q_error(input_card, true_card):
    input_card = float(input_card)
    true_card = float(true_card)
    return max([max([input_card, 1])/max([true_card, 1]), max([1, true_card])/max([1, input_card])])


def evaluation(graph_file, data_list, learned_model, filter_method, sampler, loss_function):
    learned_model.eval()
    average_q_error = list()
    for f in data_list:
        query_graph_info = load_p_data(f.replace('.graph','.grf').replace('/query_graph/', '/grf_query_graph/'))
        query_edge_list = torch.LongTensor(query_graph_info[3]).to(args.device)
        query_feat = generate_features(query_graph_info, single_feat_dim).to(args.device)
        true_value = torch.tensor(int(baseline_dict[f]), dtype=torch.float).to(args.device)

        filter_method.update_query(query_graph_info)
        sampler.update_query(query_graph_info)
        start_time = time.time()
        candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = filter_method.cpp_GQL(f,  graph_file.replace('.grf', '.graph').replace('/grf_data_graph/', '/data_graph/'))
        filter_end_time = time.time()
        if 0 in candidate_count:
            continue
        starting_vertex = candidate_count.index(min(candidate_count))
        # new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = sampler.find_subgraph(
            # starting_vertex, candidates)
        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.load_induced_subgraph(candidates,induced_subgraph_list, neighbor_offset)

        num_subgraphs = len(new_vertices)
        output_pred = list()
        for i in range(num_subgraphs):
            subgraph_info = [new_vertices[i], new_v_label[i], new_degree[i], new_edges[i], new_e_label[i],
                             new_v_neigh[i]]
            subgraph_feat = generate_features(subgraph_info, single_feat_dim).to(args.device)
            preprocessed_edgelist = torch.LongTensor(preprocess_data_edge(subgraph_info[0], subgraph_info[3])).to(args.device)
            if args.model_name == 'attentive' or args.model_name == 'wasserstein':
                query2data_edge_list = torch.LongTensor(preprocess_query2data(subgraph_info[0], candidate_info)).to(args.device)
                pred_count,_,_ = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist, query2data_edge_list)
            elif args.model_name == 'basic':
                pred_count = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist)
            else:
                raise NotImplementedError
            output_pred.append(pred_count)
        output_pred = torch.cat(output_pred, dim=1)
        sum_pred = torch.sum(output_pred)
        compute_end_time = time.time()
        test_loss = loss_function(sum_pred, true_value)
        q_error_result = q_error(sum_pred, true_value)
        average_q_error.append(q_error_result)
    return sum(average_q_error)/len(average_q_error)


def curriculum_train(graph_file, train_data, train_model, loss_function, args, opt):
    train_model.train()
    # initialize the important parameters.
    num_epoch = args.num_epoch
    default_batch_size = args.batch_size
    flag = 0
    batch_num = 0
    half_batch_size = math.floor(0.5 * default_batch_size)
    quarter_batch_size = math.floor(0.25 * default_batch_size)
    epoch_20 = math.floor(0.1*num_epoch)
    epoch_50 = math.floor(0.3*num_epoch)
    epoch_70 = math.floor(0.7*num_epoch)
    for e_num in tqdm(range(args.num_epoch)):
        # set the batch size and the query selection criterion.
        batch_num = 0
        query_flag = 'all'
        if e_num <= epoch_20:
            batch_size = 1
            query_flag = 'one'
        elif epoch_20 <e_num <= epoch_50:
            batch_size = half_batch_size
            query_flag = 'one'
        elif epoch_50< e_num <= epoch_70:
            batch_size = half_batch_size
        elif e_num > epoch_70:
            batch_size = default_batch_size

        for f in train_data:
            query_graph_info = load_p_data(f.replace('.graph','.grf').replace('/query_graph/', '/grf_query_graph/'))
            query_edge_list = torch.LongTensor(query_graph_info[3])
            query_feat = generate_features(query_graph_info, single_feat_dim)
            true_value = torch.tensor(int(baseline_dict[f]), dtype=torch.float)
            # print(feat)
            if flag == 0:
                filter_model = Filtering(query_graph_info, data_graph_info)
                subgraph_sampler = SampleSubgraph(query_graph_info, data_graph_info)
                flag = 1
            else:
                filter_model.update_query(query_graph_info)
                subgraph_sampler.update_query(query_graph_info)
            candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = filter_model.cpp_GQL(f,  graph_file.replace('.grf', '.graph').replace('/grf_data_graph/', '/data_graph/'))
            if 0 in candidate_count:
                continue
            start_time = time.time()
            # use different sampling method.
            if args.sample_method == 'start':
                starting_vertex = candidate_count.index(min(candidate_count))
                new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph(starting_vertex, candidates)
            elif args.sample_method == 'induced':
                # new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph_induced(candidates)
                new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.load_induced_subgraph(candidates,induced_subgraph_list, neighbor_offset)
            end_time = time.time()

            num_subgraphs = len(new_vertices)
            # at the very begining, skip the query with more than 1 target subgraphs.
            if query_flag == 'one' and num_subgraphs > 1:
                continue
            output_pred = list()
            for i in range(num_subgraphs):
                subgraph_info = [new_vertices[i], new_v_label[i], new_degree[i], new_edges[i], new_e_label[i], new_v_neigh[i]]
                subgraph_feat = generate_features(subgraph_info, single_feat_dim)
                preprocessed_edgelist = torch.LongTensor(preprocess_data_edge(subgraph_info[0], subgraph_info[3]))
                pred_count = train_model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist)
                output_pred.append(pred_count)
            output_pred = torch.cat(output_pred, dim=1)
            sum_pred = torch.sum(output_pred)
            loss = loss_func(sum_pred, true_value)
            loss = loss/batch_size
            if batch_size == 1:
                loss.backward()
                opt.step()
            elif batch_num == 0:
                # loss_list = loss.unsqueeze(0)
                loss.backward()
                batch_num += 1
            elif batch_num < batch_size - 1:
                # loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                batch_num += 1
            elif batch_num == batch_size - 1:
                # loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                # mean_loss = loss_list.mean()
                # mean_loss.backward()
                loss.backward()
                opt.step()
                opt.zero_grad()
                batch_num = 0
            else:
                raise NotImplementedError('Somewhere is wrong!')
        
        # after one epoch, confirm the loss is backpropagated
        if batch_size != 1 and batch_num != 0:
            mean_loss = loss_list.mean()
            mean_loss.backward()
            opt.step()
            if e_num%5 == 0:
                print('Current train loss is:')
                print(loss_list)
        if e_num%5 == 0:
            print('Now at {}th epoch.'.format(e_num))
            print('Average q-error on test set is: {}'.format(evaluation(graph_file, test_name_list, train_model, filter_model, subgraph_sampler, loss_func)))
    
    return train_model, filter_model, subgraph_sampler



if __name__=='__main__':
    graph_file = args.file_folder + args.graph_file+'.grf'
    data_graph_info = load_g_graph(graph_file)
    query_files = os.listdir(args.query_path)
    single_feat_dim = args.in_feat//4
    baseline_file = args.baseline_path+args.graph_file+args.baseline_name
    if args.graph_file in ['yeast', 'youtube', 'eu2005', 'wordnet'] and args.query_vertex_num in ['12','16', '24', '32', 'all']:
        baseline_dict = load_iso_baseline('../iso_cardinality/'+args.graph_file+'/', args.query_path.replace('/grf_query_graph/', '/query_graph/'))
    else:
        baseline_dict = load_baseline(baseline_file)
    all_name_list = list(baseline_dict.keys())
    # get training set and test set.
    train_name_list, test_name_list = train_and_test(args.query_vertex_num, args.train_percent, all_name_list)
    flag = 0

    if args.model_name == 'basic':
        model = BasicCountNet(args.in_feat, args.hidden_dim, args.hidden_dim, args.out_dim, args.pooling_method).to(args.device)
    elif args.model_name == 'attentive':
        model = AttentiveCountNet(args.in_feat, args.hidden_dim, args.hidden_dim, args.out_dim, args.pooling_method).to(args.device)
    elif args.model_name == 'wasserstein':
        model = AttentiveCountNet(args.in_feat, args.hidden_dim, args.hidden_dim, args.out_dim, args.pooling_method).to(args.device)
        wdiscriminator = WasserstainDiscriminator(2*args.out_dim)
        optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    else:
        raise NotImplementedError('The model name is not implemented.')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    # loss_func = nn.MSELoss()
    # loss_func = QErrorLoss()
    loss_func = QErrorLikeLoss()

    if args.train_method == 'Curriculum':
        model, filter_model, subgraph_sampler = curriculum_train(graph_file, train_name_list, model, loss_func, args, optimizer)
    elif args.train_method == 'Normal':
        subgraph_dict = dict()
        for epoch_num in tqdm(range(args.num_epoch)):
            model.train()
            batch_num = 0
            for f in tqdm(train_name_list):
                optimizer.zero_grad()
                query_graph_info = load_p_data(f.replace('.graph','.grf').replace('/query_graph/', '/grf_query_graph/'))
                query_edge_list = torch.LongTensor(query_graph_info[3]).to(args.device)
                query_feat = generate_features(query_graph_info, single_feat_dim)
                query_feat.to(args.device)
                true_value = torch.tensor(int(baseline_dict[f]), dtype=torch.float)
                true_value.to(args.device)
                # print(feat)
                
                if flag == 0:
                    filter_model = Filtering(query_graph_info, data_graph_info)
                    subgraph_sampler = SampleSubgraph(query_graph_info, data_graph_info)
                    flag = 1
                else:
                    filter_model.update_query(query_graph_info)
                    subgraph_sampler.update_query(query_graph_info)
                # candidates, candidate_count = filter_model.GQL_filter()
                
                # print(candidates)
                
                if epoch_num == 0:
                    t_0 = time.time()
                    candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = filter_model.cpp_GQL(f, graph_file.replace('.grf', '.graph').replace('/grf_data_graph/', '/data_graph/'))
                    t_1 = time.time()
                    # print('filter time: {}s'.format(t_1-t_0))
                    # print(candidate_count)
                    # continue
                    if args.sample_method == 'start':
                        starting_vertex = candidate_count.index(min(candidate_count))
                        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph(starting_vertex, candidates)
                        subgraph_dict[f] = [new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh]
                    elif args.sample_method == 'induced':
                        # new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph_induced(candidates)
                        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.load_induced_subgraph(candidates,induced_subgraph_list, neighbor_offset)
                        subgraph_dict[f] = [new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh]
                    else:
                        raise NotImplementedError('Sample method {} is not supported'.format(args.sample_method))
                    t_2 = time.time()
                    # print('building time: {}s'.format(t_2-t_1))
                else:
                    new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_dict[f]
                    if args.model_name == 'attentive' or args.model_name == 'wasserstein':
                        candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = filter_model.cpp_GQL(f, graph_file.replace('.grf', '.graph').replace('/grf_data_graph/', '/data_graph/'))
                # if 0 in candidate_count:
                #     print(graph_file)
                #     continue
                start_time = time.time()
                
                end_time = time.time()        

                num_subgraphs = len(new_vertices)
                # if num_subgraphs>1:
                # print(num_subgraphs)
                output_pred = list()
                for i in range(num_subgraphs):
                    subgraph_info = [new_vertices[i], new_v_label[i], new_degree[i], new_edges[i], new_e_label[i], new_v_neigh[i]]
                    subgraph_feat = generate_features(subgraph_info, single_feat_dim)
                    subgraph_feat.to(args.device)
                    preprocessed_edgelist = torch.LongTensor(preprocess_data_edge(subgraph_info[0], subgraph_info[3])).to(args.device)

                    # preprocessed_edgelist = preprocess_data_edge(subgraph_info[0], subgraph_info[3])
                    # e_u = preprocessed_edgelist[0]
                    # e_v = preprocessed_edgelist[1]
                    # preprocessed_e_u = torch.cuda.LongTensor()
                    # preprocessed_e_v = torch.cuda.LongTensor()
                    # for u in e_u:
                    #     preprocessed_e_u = torch.cat((preprocessed_e_u, torch.cuda.LongTensor(u)))
                    # for v in e_v:
                    #     preprocessed_e_v = torch.cat((preprocessed_e_v, torch.cuda.LongTensor(v)))
                    # preprocessed_edgelist = torch.cat((preprocessed_e_u.unsqueeze(0), preprocessed_e_v.unsqueeze(0)), dim=0)
                    # preprocessed_edgelist.to(args.device)

                    # print(preprocessed_edgelist.device())
                    # print(preprocessed_edgelist)
                    if args.model_name == 'attentive':
                        query2data_edge_list = torch.LongTensor(preprocess_query2data(subgraph_info[0], candidate_info)).to(args.device)
                        pred_count,_,_ = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist, query2data_edge_list)
                    elif args.model_name == 'basic':
                        pred_count = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist)
                    elif args.model_name == 'wasserstein':
                        query2data_edge_list = torch.LongTensor(preprocess_query2data(subgraph_info[0], candidate_info)).to(args.device)
                        pred_count, out_query_x, out_data_x = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist, query2data_edge_list)
                        for j in range(args.dis_iter_num):
                            wq = wdiscriminator(out_query_x)
                            wd = wdiscriminator(out_data_x)
                            anchor_q = wq.view(-1).argsort(descending=False)[:wq.size(0)].clone()
                            anchor_d = wd.view(-1).argsort(descending=True)[:wq.size(0)].clone()
                            out_query_x_anchored = out_query_x[anchor_q, :].clone().detach()
                            out_data_x_anchored = out_data_x[anchor_d, :].clone().detach()
                            optimizer_wd.zero_grad()
                            wd_loss = -torch.mean(wdiscriminator(out_query_x_anchored)) + torch.mean(wdiscriminator(out_data_x_anchored))
                            wd_loss.backward()
                            optimizer_wd.step()
                            for p in wdiscriminator.parameters():
                                p.data.clamp_(-0.01,0.01)
                        # wq = wdiscriminator(out_query_x)
                        # wd = wdiscriminator(out_data_x)
                        # anchor_q = wq.view(-1).argsort(descending=False)[:wq.size(0)]
                        # anchor_d = wd.view(-1).argsort(descending=True)[:wq.size(0)]
                        # out_query_x_anchored = out_query_x[anchor_q, :]
                        # out_data_x_anchored = out_data_x[anchor_d, :]
                        # if i == 0:
                        #     wd_loss_out = -torch.mean(wdiscriminator(out_query_x_anchored).clone()) + torch.mean(wdiscriminator(out_data_x_anchored).clone())
                        # elif i == num_subgraphs -1:
                        #     wd_loss_out += -torch.mean(wdiscriminator(out_query_x_anchored).clone()) + torch.mean(wdiscriminator(out_data_x_anchored).clone())
                        #     wd_loss_out /= num_subgraphs
                        # else:
                        #     wd_loss_out += -torch.mean(wdiscriminator(out_query_x_anchored).clone()) + torch.mean(wdiscriminator(out_data_x_anchored).clone())
                        
                    else:
                        raise NotImplementedError
                    output_pred.append(pred_count)
                output_pred = torch.cat(output_pred, dim=1)
                t_3 = time.time()
                # print('GNN time: {}s'.format(t_3-t_2))
                # print(output_pred)
                sum_pred = torch.sum(output_pred)

                # if args.model_name == 'wasserstein':
                #     wq = wdiscriminator(out_query_x)
                #     anchor_q = wq.view(-1).argsort(descending=False)[:wq.size(0)].clone()
                #     out_query_x_anchored = out_query_x[anchor_q, :].clone()
                #     wd_loss_out = -torch.mean(wdiscriminator(out_query_x_anchored).clone())

                loss = loss_func(sum_pred, true_value)
                if args.model_name == 'wasserstein':
                    if batch_num == 0:
                        # update_wd_loss = wd_loss_out.unsqueeze(0)
                        out_query_x_loss = out_query_x
                        loss_list = loss.unsqueeze(0)
                        batch_num += 1
                    elif batch_num < args.batch_size - 1:
                        loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                        out_query_x_loss = torch.cat((out_query_x_loss, out_query_x))
                        # update_wd_loss = torch.cat((update_wd_loss, wd_loss_out.unsqueeze(0)))
                        batch_num += 1
                    elif batch_num == args.batch_size - 1:
                        loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                        out_query_x_loss = torch.cat((out_query_x_loss, out_query_x))
                        wd_loss_all = -torch.mean(wdiscriminator(out_query_x_loss).clone())
                        # update_wd_loss = torch.cat((update_wd_loss, wd_loss_out.unsqueeze(0)))
                        mean_loss = loss_list.mean()
                        # wd_loss_all = update_wd_loss.mean()
                        mean_loss_update = (1-args.alpha)*wd_loss_all + args.alpha*mean_loss
                        mean_loss_update.backward()
                        optimizer.step()
                        batch_num = 0
                    else:
                        raise NotImplementedError('Somewhere is wrong!')
                else:
                    if batch_num == 0:
                        loss_list = loss.unsqueeze(0)
                        batch_num += 1
                    elif batch_num < args.batch_size - 1:
                        loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                        batch_num += 1
                    elif batch_num == args.batch_size - 1:
                        loss_list = torch.cat((loss_list, loss.unsqueeze(0)))
                        mean_loss = loss_list.mean()
                        mean_loss.backward()
                        optimizer.step()
                        batch_num = 0
                    else:
                        raise NotImplementedError('Somewhere is wrong!')
            # after one epoch, confirm the loss is backpropagated
            if batch_num != 0:
                mean_loss = loss_list.mean()
                # wd_loss_all = update_wd_loss.mean()
                
                if args.model_name == 'wasserstein':
                    wd_loss_all = -torch.mean(wdiscriminator(out_query_x_loss).clone())
                    mean_loss_update = (1-args.alpha)*wd_loss_all + args.alpha*mean_loss
                else:
                    mean_loss_update = mean_loss
                mean_loss_update.backward()
                optimizer.step()
            if epoch_num%5 == 0:
                print('Now at {}th epoch.'.format(epoch_num))
                print('Average q-error on test set is: {}'.format(evaluation(graph_file, test_name_list, model, filter_model, subgraph_sampler, loss_func)))
                # evaluation()
    else:
        raise NotImplementedError('The training method {} is not supported'.format(args.train_method))

    # save model
    torch.save(model.state_dict(), model_save_path+model_save_name)

    # start evaluation
    model.eval()
    for f in test_name_list:
        query_graph_info = load_p_data(f.replace('.graph','.grf').replace('/query_graph/', '/grf_query_graph/'))
        query_edge_list = torch.LongTensor(query_graph_info[3]).to(args.device)
        query_feat = generate_features(query_graph_info, single_feat_dim).to(args.device)
        true_value = torch.tensor(int(baseline_dict[f]), dtype=torch.float)

        filter_model.update_query(query_graph_info)
        subgraph_sampler.update_query(query_graph_info)
        start_time = time.time()
        # candidates, candidate_count = filter_model.GQL_filter()
        candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info = filter_model.cpp_GQL(f, graph_file.replace('.grf', '.graph').replace('/grf_data_graph/', '/data_graph/'))
        filter_end_time = time.time()
        if 0 in candidate_count:
            print('{} shoule have 0 subgraph in data graph.'.format(f))
            continue
        
        if args.sample_method == 'start':
            starting_vertex = candidate_count.index(min(candidate_count))
            new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph(starting_vertex, candidates)
        elif args.sample_method == 'induced':
            # new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.find_subgraph_induced(candidates)
            new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = subgraph_sampler.load_induced_subgraph(candidates,induced_subgraph_list, neighbor_offset)
        else:
            raise NotImplementedError('Sample method {} is not supported'.format(args.sample_method))

        num_subgraphs = len(new_vertices)
        output_pred = list()
        for i in range(num_subgraphs):
            subgraph_info = [new_vertices[i], new_v_label[i], new_degree[i], new_edges[i], new_e_label[i],
                             new_v_neigh[i]]
            subgraph_feat = generate_features(subgraph_info, single_feat_dim).to(args.device)
            preprocessed_edgelist = torch.LongTensor(preprocess_data_edge(subgraph_info[0], subgraph_info[3])).to(args.device)
            if args.model_name == 'attentive' or args.model_name == 'wasserstein':
                query2data_edge_list = torch.LongTensor(preprocess_query2data(subgraph_info[0], candidate_info)).to(args.device)
                pred_count,_,_ = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist, query2data_edge_list)
            elif args.model_name == 'basic':
                pred_count = model(query_feat, subgraph_feat, query_edge_list, preprocessed_edgelist)
            else:
                raise NotImplementedError
            output_pred.append(pred_count)
        output_pred = torch.cat(output_pred, dim=1)
        sum_pred = torch.sum(output_pred)
        compute_end_time = time.time()
        test_loss = loss_func(sum_pred, true_value)
        q_error_result = q_error(sum_pred, true_value)
        print(f)
        print(test_loss)
        print(q_error_result)
        with open(result_save_path+result_save_name, 'a') as f1:
            f1.write(f + ' ' + str(q_error_result) + ' ' + str(float(sum_pred))+ ' ' + str(float(true_value)) + ' '
                    + str(filter_end_time - start_time)+ ' ' + str(compute_end_time - start_time) +'\n')
