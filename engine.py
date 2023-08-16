import torch
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
import torch_geometric
from torchmetrics import R2Score

ModelEma = ModelEmaV2


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_data(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list,
                    x,y,batch,edge_occu,
                    edge_src,edge_dst,edge_vec,edge_attr,edge_num,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100,
                    logger=None):
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    r2_metric = AverageMeter()


    start_time = time.perf_counter()

    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std


    with amp_autocast():
        pred = model(batch=batch,
            edge_occu=edge_occu,f_in=x, edge_src=edge_src, edge_dst=edge_dst,edge_attr=edge_attr,
            edge_vec=edge_vec, edge_num = edge_num)

        pred = pred.squeeze()
        loss = criterion(pred, (y - task_mean) / task_std)
    optimizer.zero_grad()
    if loss_scaler is not None:
        loss_scaler(loss, optimizer, parameters=model.parameters())
    else:
        loss.backward()
        if clip_grad is not None:
            dispatch_clip_grad(model.parameters(),
                value=clip_grad, mode='norm')
        optimizer.step()
    y_pred = pred.detach() * task_std + task_mean
    y_true = y

    mae_metric.update(torch.mean(torch.abs(y_pred-y_true)).item(), n=1)
    #r2score = R2Score().to(device)
    #r2_metric.update((r2score(y_pred,y_true)),n=pred.shape[0])

    return mae_metric.avg



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list, 
                    target: int,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    model_ema: Optional[ModelEma] = None,  
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100, 
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    r2_metric = AverageMeter()


    start_time = time.perf_counter()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std

    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        with amp_autocast():
            pred = model(batch=data.batch, 
                edge_occu=data.edge_occu,f_in=data.x, edge_src=data.edge_src, edge_dst=data.edge_dst,edge_attr=data.edge_attr,
                edge_vec=data.edge_vec, edge_num = data.edge_num)

            pred = pred.squeeze()
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            optimizer.step()
        
        loss_metric.update(loss.item(), n=pred.shape[0])
        y_pred = pred.detach() * task_std + task_mean
        y_true = data.y[:, target]

        mae_metric.update(torch.mean(torch.abs(y_pred-y_true)).item(), n=pred.shape[0])
        #r2score = R2Score().to(device)
        #r2_metric.update((r2score(y_pred,y_true)),n=pred.shape[0])


        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: #time.perf_counter() - wall_print > 15:
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = 'Epoch: [{epoch}][{step}/{length}] loss: {loss:.5f}, MAE: {mae:.5f}, time/step={time_per_step:.0f}ms, '.format( 
                epoch=epoch, step=step, length=len(data_loader), 
                mae=mae_metric.avg, 
                loss=loss_metric.avg,
                time_per_step=(1e3 * w / e / len(data_loader))
                )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
        
    return mae_metric.avg


def evaluate(model, norm_factor, target, data_loader, device, amp_autocast=None, 
    print_freq=100, logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    r2_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std
    
    with torch.no_grad():
            
        for data in data_loader:
            data = data.to(device)
            with amp_autocast():
                pred = model(batch=data.batch, 
                    edge_occu=data.edge_occu,f_in=data.x,edge_src=data.edge_src,edge_dst=data.edge_dst, edge_attr=data.edge_attr,
                    edge_vec=data.edge_vec, edge_num = data.edge_num)
                pred = pred.squeeze()
            
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            loss_metric.update(loss.item(), n=pred.shape[0])
            y_pred = pred.detach() * task_std + task_mean
            y_true = data.y[:, target]
                
            mae_metric.update(torch.mean(torch.abs(y_pred-y_true)).item(), n=pred.shape[0])
            #r2score = R2Score().to(device)
            #r2_metric.update((r2score(y_pred,y_true)),n=pred.shape[0])


    return mae_metric.avg, loss_metric.avg


def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)
