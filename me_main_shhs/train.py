from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pl_bolts.models.regression import LogisticRegression
import pytorch_lightning as pl
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from pytorch_lightning.callbacks import EarlyStopping
from copy import deepcopy

import time
import logging
import warnings

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")



# Train, test
def evaluate(q_encoder, train_loader, test_loader,device, i):

    # eval
    q_encoder.eval()

    # process val
    emb_val, gt_val = [], []

    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.float()
            y_val = y_val.long()
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val, proj="mid").cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test, gt_test = [], []

    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.float()
            y_test = y_test.long()
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test, proj="mid").cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())

    emb_test, gt_test = np.array(emb_test), np.array(gt_test)

    acc, cm, f1, kappa, bal_acc, gt, pd = task(emb_val, emb_test, gt_val, gt_test,i)

    q_encoder.train()
    return acc, cm, f1, kappa, bal_acc, gt, pd


def task(X_train, X_test, y_train, y_test,i):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    start = time.time()
    
    train = SklearnDataset(X_train, y_train)  
    train = DataLoader(train, batch_size=256, shuffle=True)
    class LinModel(LogisticRegression): 
        
        def training_epoch_end(self, outputs):
            epoch_loss = torch.hstack([x['loss'] for x in outputs]).mean()
            self.log("epoch_loss", epoch_loss)
            
    model = LinModel(input_dim=256, num_classes=5)
    
    early_stop_callback = EarlyStopping(monitor="epoch_loss", min_delta=0.001, patience= 5, mode="min", verbose=False)
    lin_trainer = pl.Trainer(callbacks=[early_stop_callback], gpus=1, precision=16, num_sanity_val_steps=0, enable_checkpointing=False, max_epochs=500, auto_lr_find=True)
    lin_trainer.fit(model, train)
    pred = model(torch.Tensor(X_test)).detach().cpu().numpy()
    pred = np.argmax(pred, axis = 1)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    kappa = cohen_kappa_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    
    pit = time.time() - start
    print(f"Took {int(pit // 60)} min:{int(pit % 60)} secs for {i} fold")

    return acc, cm, f1, kappa, bal_acc, y_test, pred

def kfold_evaluate(q_encoder, test_subjects,device, BATCH_SIZE):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

    total_acc, total_f1, total_kappa, total_bal_acc = [], [], [], []
    i = 1
    
    
    for train_idx, test_idx in kfold.split(test_subjects):

        test_subjects_train = [test_subjects[i] for i in train_idx]
        test_subjects_test = [test_subjects[i] for i in test_idx]

        train_loader = DataLoader(TuneDataset(test_subjects_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TuneDataset(test_subjects_test), batch_size=BATCH_SIZE, shuffle= False)
        test_acc, _, test_f1, test_kappa, bal_acc, gt, pd = evaluate(q_encoder, train_loader, test_loader,device, i)

        total_acc.append(test_acc)
        total_f1.append(test_f1)
        total_kappa.append(test_kappa)
        total_bal_acc.append(bal_acc)
        
        print("+"*50)
        print(f"Fold{i} acc: {test_acc}")
        print(f"Fold{i} f1: {test_f1}")
        print(f"Fold{i} kappa: {test_kappa}")
        print(f"Fold{i} bal_acc: {bal_acc}")
        print("+"*50)
        i+=1 

    return np.mean(total_acc), np.mean(total_f1), np.mean(total_kappa), np.mean(total_bal_acc)

class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index]
        y =  self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]
        
    def _add_subjects(self):
        self.X = []
        self.y = []
        for subject in self.subjects:
            self.X.append(np.expand_dims(subject['x'][:,0],axis=1))
            self.y.append(subject['y'])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)



##################################################################################################################################################


# Pretrain
def Pretext(
    q_encoder,
    optimizer,
    Epoch,
    criterion,
    pretext_loader,
    test_subjects,
    wandb,
    device, 
    SAVE_PATH,
    BATCH_SIZE
):


    step = 0
    best_f1 = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(Epoch):
        pretext_loss = []        
        print('=========================================================\n')
        print("Epoch: {}".format(epoch))
        print('=========================================================\n')
        all_loss = []
        for index, (aug1, aug2) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):
            q_encoder.train()
            aug1 = aug1.float()
            aug2 = aug2.float()

            aug1, aug2 = (
                aug1.to(device),
                aug2.to(device),
            )  # (B, 7, 2, 3000)  (B, 7, 2, 3000) (B, 7, 2, 3000)
        
            with torch.cuda.amp.autocast():           
                
                num_len = aug1.shape[1]
                aug1_features = []
                aug2_features = []
                
                for i in range(num_len):
                    aug1_features.append(q_encoder(aug1[:, i], proj='top')) #(B, 128)
                    aug2_features.append(q_encoder(aug2[:, i], proj='top'))  # (B, 128)

                aug1_features = torch.stack(aug1_features, dim=1)  # (B, 7, 128)
                aug2_features = torch.stack(aug2_features, dim=1)  # (B, 7, 128)
                               
                sel1 = np.random.choice(num_len)
                sel2 = np.random.choice(num_len)       
                loss1 = criterion(aug1_features[:, sel1], aug1_features)
                loss2 = criterion(aug2_features[:, sel2], aug2_features)
                    
                loss = (loss1 + loss2)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss back
            all_loss.append(loss.item())
            pretext_loss.append(loss.cpu().detach().item())


        scheduler.step(sum(all_loss[-50:]))
        lr = optimizer.param_groups[0]["lr"]
        wandb.log({"ssl_lr": lr, "Epoch": epoch})

        wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})

        if epoch >=20 and (epoch) % 5 == 0:
            
            test_acc, test_f1, test_kappa, bal_acc = kfold_evaluate(
                q_encoder, test_subjects, device, BATCH_SIZE
            )

            wandb.log({"Valid Acc": test_acc, "Epoch": epoch})
            wandb.log({"Valid F1": test_f1, "Epoch": epoch})
            wandb.log({"Valid Kappa": test_kappa, "Epoch": epoch})
            wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": epoch})

            if test_f1 > best_f1:   
                best_f1 = test_f1
                torch.save(q_encoder.enc.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                print("save best model on test set with best F1 score")


