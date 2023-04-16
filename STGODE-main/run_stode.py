import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from model import ODEGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np
import xlwt


def train(loader, model, optimizer, criterion, device):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
    return batch_loss / (idx + 1)


# @torch.no_grad()
# def eval(loader, model, std, mean, device):
#     batch_rmse_loss = 0
#     batch_mae_loss = 0
#     batch_mape_loss = 0
#     for idx, (inputs, targets) in enumerate(tqdm(loader)):
#         model.eval()
#
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#         output = model(inputs)
#
#         out_unnorm = output.detach().cpu().numpy()*std + mean
#         target_unnorm = targets.detach().cpu().numpy()
#
#         mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
#         rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
#         mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
#         batch_rmse_loss += rmse_loss
#         batch_mae_loss += mae_loss
#         batch_mape_loss += mape_loss
#
#     return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0
    batch_mae_loss = 0
    batch_mape_loss = 0
    pre = []
    lab = []
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)

        out_unnorm = output.detach().cpu().numpy() * std + mean
        target_unnorm = targets.detach().cpu().numpy()
        pre.append(out_unnorm)
        lab.append(target_unnorm)

    input = np.concatenate(lab, 0)
    prediction = np.concatenate(pre, 0)  # (batch, T', 1)

    # NODE_id = 9
    #
    # pre = []
    # lab = []
    #
    # for i in range(prediction.shape[0]):
    #     # if i % 11 == 0 or i ==0:
    #     pre.append(prediction[i, NODE_id, :-1])
    #     lab.append(input[i, NODE_id,:-1])
    # pre = np.concatenate(pre, 0)  # (batch', 1)
    # lab = np.concatenate(lab, 0)  # (batch', 1)
    # workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet = workbook.add_sheet('My Worksheet')
    # worksheet.write(0, 0, "target")
    # worksheet.write(0, 1, "prediction")
    # print("pre:{pre},  lab:{lab}".format(pre=pre.shape, lab=lab.shape))
    # for j in range(pre.shape[0]):
    #     worksheet.write(j + 1, 0, label=str(lab[j]))
    #     worksheet.write(j + 1, 1, label=str(pre[j]))
    #
    # # 保存
    # # workbook.save('PeMS04_single_head.xls')
    # workbook.save('my_Pems04.xls')


    print("input:", input.shape)
    print("prediction:", prediction.shape)
    # 计算误差
    excel_list = []
    prediction_length = prediction.shape[2]
    metric_method = 'mask'
    for i in range(prediction_length):
        mae = masked_mae_np(input[:, :, i], prediction[:, :, i], 0.0)
        rmse = masked_rmse_np(input[:, :, i], prediction[:, :, i], 0.0)
        mape = masked_mape_np(input[:, :, i], prediction[:, :, i], 0)
        print('MAE: %.2f' % (mae))
        print('RMSE: %.2f' % (rmse))
        print('MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])

    # print overall results
    mae = masked_mae_np(input.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
    rmse = masked_rmse_np(input.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
    mape = masked_mape_np(input.reshape(-1, 1), prediction.reshape(-1, 1), 0)

    print('all MAE: %.2f' % (mae))
    print('all RMSE: %.2f' % (rmse))
    print('all MAPE: %.2f' % (mape))
    excel_list.extend([mae, rmse, mape])
    print(excel_list)

def main(args):
    # random seed
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    USE_CUDA = torch.cuda.is_available()
    #
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
    #
    # print("CUDA:", USE_CUDA, DEVICE)

    device = torch.device('cuda:'+str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')
    print("CUDA:", USE_CUDA, device)
    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    data, mean_, std_, dtw_matrix, sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader, mean, std = generate_dataset(data, mean_, std_, args)
    A_sp_wave = get_normalized_adj(sp_matrix).to(device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(device)

    net = ODEGCN(num_nodes=data.shape[1], 
                num_features=data.shape[2], 
                num_timesteps_input=args.his_length, 
                num_timesteps_output=args.pred_length, 
                A_sp_hat=A_sp_wave, 
                A_se_hat=A_se_wave)
    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    # best_valid_rmse = 1000
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    #
    # for epoch in range(1, args.epochs+1):
    #     print("=====Epoch {}=====".format(epoch))
    #     print('Training...')
    #     loss = train(train_loader, net, optimizer, criterion, device)
    #     print('Evaluating...')
    #     train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, device)
    #     valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, device)
    #
    #     if valid_rmse < best_valid_rmse:
    #         best_valid_rmse = valid_rmse
    #         print('New best results!')
    #         torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')
    #
    #     if args.log:
    #         logger.info(f'\n##on train data## loss: {loss}, \n' +
    #                     f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
    #                     f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
    #     else:
    #         print(f'\n##on train data## loss: {loss}, \n' +
    #             f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
    #             f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
    #
    #     scheduler.step()

    net.load_state_dict(torch.load(f'net_params_{args.filename}_{0}.pkl'))
    test_rmse, test_mae, test_mape = eval(test_loader, net, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')


if __name__ == '__main__':
    main(args)
