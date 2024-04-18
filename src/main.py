import random
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from data_loader import Dataset
from logger import get_args, logger
from models import MFModel, BoxModel
from utils import *


def main():
    args, parser = get_args()
    logger.set_log_file(args, parser)
    logger.print(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    dataset = Dataset(args.dataset, device=device)
    logger.print(dataset)

    
    model = MFModel(dataset, n_hidden=args.n_hidden, device=device)

    if args.box:
        model = BoxModel(
            model, n_hidden=args.n_hidden_box, attn=args.attn,
            mask=args.mask, tau=args.tau, beta=args.beta, gd=args.gd,
            device=device)
    logger.print(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_data = []    
    for uid, (pos_iids, _) in dataset.train_data.items():
        for pos_iid in pos_iids:
            neg_iid = random.randint(0, dataset.n_items - 1)
            train_data.append([uid, pos_iid, neg_iid, dataset.item2cid[pos_iid], dataset.item2cid[neg_iid]])

    bs = args.bs
    best_score, best_model, best_epoch, patience = 0, None, 0, 0
    
    for epoch in range(1, args.ne + 1):
        logger.print(f'[Epoch {epoch}]')
        random.shuffle(train_data)
        
        tqdm_batch = tqdm(range(0, len(train_data), bs), bar_format='{l_bar}{r_bar}', desc='[Training]')

        total_rec_loss, total_box_reg_loss, total_mask_reg_loss = 0, 0, 0
        for step, i in enumerate(tqdm_batch, start=1):
            batch_data = torch.LongTensor(train_data[i:i + bs]).to(device)
            pos_out = model.forward(batch_data[:, 0], batch_data[:, 1])
            neg_out = model.forward(batch_data[:, 0], batch_data[:, 2])
            pos_scores = pos_out[0]
            neg_scores = neg_out[0]
            
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores))) * args.rec

            total_rec_loss += loss.item()
            if not args.box:
                user_embeddings, pos_item_embeddings = pos_out[1], pos_out[2]
                neg_item_embeddings = neg_out[2]
                reg_loss = (torch.norm(user_embeddings) ** 2
                            + torch.norm(pos_item_embeddings) ** 2
                            + torch.norm(neg_item_embeddings) ** 2) / 2
                reg_loss = args.ler * reg_loss / user_embeddings.size(0)
                total_box_reg_loss += reg_loss.item()
                loss += reg_loss
                
            if args.box:
                offsets = torch.concat([pos_out[1], pos_out[2], neg_out[1], neg_out[2]])
                box_reg_loss = torch.abs(torch.mean(
                    offsets - args.eta_box, dim=1)).mean() * args.lbr
                total_box_reg_loss += box_reg_loss.item()
                loss += box_reg_loss
                
            if args.box and args.mask:
                masks = torch.concat([pos_out[3], neg_out[3]], dim=0)
                mask_reg_loss = torch.abs(torch.mean(
                    masks - args.eta_uc, dim=1)).mean() * args.lmr
                total_mask_reg_loss += mask_reg_loss.item()
                loss += mask_reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.ri == 0:
                tqdm_batch.write(
                    f'[Step {step}] loss = {(total_rec_loss + total_box_reg_loss + total_mask_reg_loss) / args.ri}')
                total_rec_loss, total_box_reg_loss, total_mask_reg_loss = 0, 0, 0
                

        tqdm_batch.close()

        valid_scores = valid_model(
            model, dataset.valid_data, dataset, args, metrics=[
                'ndcg@20', 'recall@20'],
            diversity=False)
        
        score = valid_scores['ndcg@20']
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            best_model = deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= args.patience:
                logger.print(
                    f'[ ][Epoch {epoch}] {valid_scores}, best = {best_score}, patience = {patience}/{args.patience}')
                logger.print('[!!! Early Stop !!!]')
                break

        logger.print(f'[{"*" if patience == 0 else " "}][Epoch {epoch}] {valid_scores}, '
                     f'best = {best_score}, patience = {patience}/{args.patience}')

    if best_model is not None:
        model.load_state_dict(best_model)
        logger.print(f'[Epoch] Total = {epoch}, Best = {best_epoch}')

    res = valid_model(
        model, dataset.test_data, dataset, args,
        metrics=['ndcg@20', 'recall@20'],
        diversity=True)

    print_results(res)
    logger.print(f'[Epoch] Total = {epoch}, Best = {best_epoch}')


if __name__ == '__main__':
    main()
