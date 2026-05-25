from import_library import *
from config import *
from early_stopping import *
from focal_loss import *
from model_architecture import *

#Threshold search
def find_optimal_threshold(y_true, y_prob, min_recall=0.70, beta=2.0):
    thresholds = np.arange(0.01, 0.95, 0.005)
    best_thresh_f1, best_f1, best_rec_f1 = 0.5, 0.0, 0.0
    best_thresh_fb, best_fb = 0.5, 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec   = recall_score(y_true, preds, zero_division=0)

        if rec >= min_recall:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh_f1, best_rec_f1 = f1, t, rec
        else:
            fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
            if fb > best_fb:
                best_fb, best_thresh_fb = fb, t

    if best_f1 > 0:
        return best_thresh_f1, best_f1, "f1_constrained"
    else:
        return best_thresh_fb, best_fb, "fbeta_fallback"

#Fraud-neighbor features
def compute_fraud_neighbor_features(edge_index_np: np.ndarray, y: np.ndarray, N: int):
    src, dst = edge_index_np[0], edge_index_np[1]
    is_fraud = (y == 1).astype(np.float32)

    fraud_nb_count = np.zeros(N, dtype=np.float32)
    total_nb_count = np.zeros(N, dtype=np.float32)

    np.add.at(fraud_nb_count, dst, is_fraud[src])
    np.add.at(total_nb_count, dst, 1.0)

    fraud_nb_ratio = fraud_nb_count / (total_nb_count + 1e-9)
    return fraud_nb_count, fraud_nb_ratio

#Extend data features
def engineer_features(feat_df: pd.DataFrame):
    df = feat_df.copy()

    #Calculate ratios
    df['sent_received_ratio'] = df['total_sent']  / (df['total_received'] + 1e-9)
    df['degree_ratio'] = df['out_degree']  / (df['in_degree'] + 1e-9)
    df['avg_sent'] = df['total_sent']  / (df['out_degree'] + 1e-9)
    df['avg_received'] = df['total_received'] / (df['in_degree'] + 1e-9)
    df['unique_ratio'] = df['unique_senders'] / (df['unique_receivers'] + 1e-9)

    #Log transforms
    for col in ['total_sent', 'total_received', 'avg_sent', 'avg_received']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    feat_cols = [
        'in_degree', 'out_degree', 'unique_senders', 'unique_receivers',
        'total_sent', 'total_received',
        'sent_received_ratio', 'degree_ratio',
        'avg_sent', 'avg_received', 'unique_ratio',
        'log_total_sent', 'log_total_received',
        'log_avg_sent',   'log_avg_received',
    ]
    return df, feat_cols

#Training
def train_gnn_model():
    config = Config()
    os.makedirs(config.OUT_DIR, exist_ok=True)
    set_seed(config.SEED)

    def load_folder(folder):
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    edge_df = load_folder(config.EDGE_DIR)
    feat_df = load_folder(config.FEAT_DIR)
    node_df = load_folder(config.NODE_DIR)

    feat_df = feat_df.drop_duplicates(subset=['account_id'], keep='last').reset_index(drop=True)
    node_df = node_df.drop_duplicates(subset=['account_id'], keep='last').reset_index(drop=True)

    #Build index
    all_acc = pd.unique(pd.concat([
        node_df['account_id'], feat_df['account_id'],
        edge_df['src'], edge_df['dst']
    ], ignore_index=True))

    account_to_idx = {a: i for i, a in enumerate(all_acc)}
    N = len(account_to_idx)

    #Extend features
    feat_df, FEAT_COLS = engineer_features(feat_df)
    n_base = len(FEAT_COLS)

    #Categorical encoders
    le_bank = LabelEncoder().fit(node_df['bank_id'].astype(str))
    le_entity = LabelEncoder().fit(node_df['entity_id'].astype(str))

    #Allocate feature matrix: base + bank + entity + 2 fraud-neighbour cols
    n_extra = 2 + 2   # bank_id, entity_id, fraud_nb_count, fraud_nb_ratio
    X = np.zeros((N, n_base + n_extra), dtype=np.float32)

    feat_df['idx'] = feat_df['account_id'].map(account_to_idx)
    valid_feat = feat_df.dropna(subset=['idx'])
    X[valid_feat['idx'].values.astype(int), :n_base] = (
        valid_feat[FEAT_COLS].fillna(0).values.astype(np.float32)
    )

    node_df['idx'] = node_df['account_id'].map(account_to_idx)
    valid_node = node_df.dropna(subset=['idx'])
    ni = valid_node['idx'].values.astype(int)

    X[ni, n_base] = le_bank.transform(valid_node['bank_id'].astype(str))
    X[ni, n_base + 1] = le_entity.transform(valid_node['entity_id'].astype(str))

    #Label
    y = np.full(N, -1, dtype=np.int64)
    y[ni] = valid_node['label'].values.astype(np.int64)

    #Edges (bidirectional)
    edge_df['src_idx'] = edge_df['src'].map(account_to_idx)
    edge_df['dst_idx'] = edge_df['dst'].map(account_to_idx)
    edge_df = edge_df.dropna(subset=['src_idx', 'dst_idx']).copy()
    edge_df['src_idx'] = edge_df['src_idx'].astype(int)
    edge_df['dst_idx'] = edge_df['dst_idx'].astype(int)

    le_pf = LabelEncoder().fit(edge_df['payment_format'].astype(str))

    edge_agg = (edge_df.groupby(['src_idx', 'dst_idx'], sort=False)
                       .agg(total_amount=('amount', 'sum'),
                            tx_count=('amount', 'count'))
                       .reset_index())

    ei_np = edge_agg[['src_idx', 'dst_idx']].values.T    
    ei_t = torch.tensor(ei_np, dtype=torch.long)
    edge_index = torch.cat([ei_t, ei_t.flip(0)], dim=1)  

    #Fraud-neighbour features
    ei_bidir = edge_index.numpy()      
    fnc, fnr = compute_fraud_neighbor_features(ei_bidir, y, N)
    X[:, n_base + 2] = fnc
    X[:, n_base + 3] = fnr

    #Scale feature
    scaler = StandardScaler()
    #Scale all continuous columns (everything except the two label-encoded cats)
    cont_cols = list(range(n_base)) + [n_base + 2, n_base + 3]
    X[:, cont_cols] = scaler.fit_transform(X[:, cont_cols])

    #Train/Val/Test split
    labeled_idx = np.where(y >= 0)[0]
    labeled_y = y[labeled_idx]

    tr_i, tmp_i, tr_y, tmp_y = train_test_split(
        labeled_idx, labeled_y,
        test_size=0.30, stratify=labeled_y, random_state=config.SEED
    )
    va_i, te_i, va_y, te_y = train_test_split(
        tmp_i, tmp_y,
        test_size=0.50, stratify=tmp_y, random_state=config.SEED
    )

    #Oversample fraud nodes
    fraud_tr_idx = tr_i[tr_y == 1]
    normal_tr_idx = tr_i[tr_y == 0]
    oversampled = np.tile(fraud_tr_idx, config.FRAUD_OVERSAMPLE)
    balanced_tr_i = np.concatenate([normal_tr_idx, oversampled])
    np.random.shuffle(balanced_tr_i)


    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long)
    )

    common_loader_kwargs = dict(
        num_neighbors=config.NUM_NEIGHBORS,
        num_workers=config.NUM_WORKERS,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    train_loader = NeighborLoader(
        data,
        batch_size=config.TRAIN_BATCH_SIZE,
        input_nodes=torch.from_numpy(balanced_tr_i),
        shuffle=True,
        **common_loader_kwargs
    )
    val_loader = NeighborLoader(
        data,
        batch_size=config.INF_BATCH_SIZE,
        input_nodes=torch.from_numpy(va_i),
        shuffle=False,
        **common_loader_kwargs
    )
    test_loader = NeighborLoader(
        data,
        batch_size=config.INF_BATCH_SIZE,
        input_nodes=torch.from_numpy(te_i),
        shuffle=False,
        **common_loader_kwargs
    )

    #Class-weight: inverse of fraud frequency in original (unbalanced) train
    normal_n = int((tr_y == 0).sum())
    fraud_n_tr = int((tr_y == 1).sum())
    pos_weight = normal_n / fraud_n_tr if config.USE_CLASS_WEIGHT else 1.0

    model = HybridGNN(
        in_dim=data.x.size(1),
        hidden=config.HIDDEN,
        heads=config.HEADS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WD
    )

    #OneCycleLR: cosine annealing with 10% linear warm-up
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.10,
        anneal_strategy='cos',
    )

    loss_fn = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA, pos_weight)
    early_stop = EarlyStopping(patience=config.PATIENCE, mode='max')
    best_ap = -1.0
    best_state = None

    #Training
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0

        for batch in train_loader:
            batch = batch.to(config.DEVICE, non_blocking=True)
            bs = batch.batch_size

            optimizer.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index)[:bs]
            loss = loss_fn(out, batch.y[:bs])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()         #Per-batch with OneCycleLR

            total_loss += loss.item() * bs
            total_seeds += bs

        avg_loss = total_loss / total_seeds

        #Validate
        model.eval()
        val_probs_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.DEVICE, non_blocking=True)
                bs = batch.batch_size
                out = model(batch.x, batch.edge_index)[:bs]
                prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs_list.append(prob)
                val_labels_list.append(batch.y[:bs].cpu().numpy())

        val_probs = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        ap = average_precision_score(val_labels, val_probs)

        if ap > best_ap:
            best_ap    = ap
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if early_stop(ap, model.state_dict()):
            print(f"\nEarly stopping triggered at epoch {epoch}  "
                  f"(best AP={best_ap:.4f})")
            break

    #Threshold search on validation set
    model.load_state_dict(best_state)
    model.to('cpu').eval()

    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to('cpu')
            bs = batch.batch_size
            out = model(batch.x, batch.edge_index)[:bs]
            prob = F.softmax(out, dim=1)[:, 1].numpy()
            val_probs_list.append(prob)
            val_labels_list.append(batch.y[:bs].numpy())

    val_probs = np.concatenate(val_probs_list)
    val_labels = np.concatenate(val_labels_list)

    best_threshold, best_score, method = find_optimal_threshold(
        val_labels, val_probs,
        min_recall=config.MIN_RECALL,
        beta=config.BETA
    )
    
    #Final evaluation on test set
    test_probs_list, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to('cpu')
            bs = batch.batch_size
            out = model(batch.x, batch.edge_index)[:bs]
            prob = F.softmax(out, dim=1)[:, 1].numpy()
            test_probs_list.append(prob)
            test_labels_list.append(batch.y[:bs].numpy())

    test_probs = np.concatenate(test_probs_list)
    test_labels = np.concatenate(test_labels_list)
    test_pred = (test_probs >= best_threshold).astype(int)

    test_auc = roc_auc_score(test_labels, test_probs)
    test_ap = average_precision_score(test_labels, test_probs)
    test_rec = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)
    test_f2 = fbeta_score(test_labels, test_pred, beta=2.0, zero_division=0)

    print(f"Threshold: {best_threshold:.4f}  (method: {method})")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test F2: {test_f2:.4f}")
    print(f"Test Recall: {test_rec:.4f}")

    print(classification_report(test_labels, test_pred, digits=4))
    print(confusion_matrix(test_labels, test_pred))

    #Save model
    artifact = {
        'model_state_dict': best_state,
        'model_config': {
            'in_dim': data.x.size(1),
            'hidden': config.HIDDEN,
            'heads': config.HEADS,
            'dropout': config.DROPOUT,
            'version': config.VERSION,
        },
        'threshold': float(best_threshold),
        'threshold_method': method,
        'min_recall_target': config.MIN_RECALL,
        'account_to_idx': account_to_idx,
        'feat_cols': FEAT_COLS,
        'n_base_feats': n_base,
        'feature_scaler': scaler,
        'cont_col_indices': cont_cols,
        'le_bank': le_bank,
        'le_entity': le_entity,
        'le_payment_format': le_pf,
        'num_neighbors': config.NUM_NEIGHBORS,
        'training_metrics': {
            'best_val_ap': float(best_ap),
            'test_auc': float(test_auc),
            'test_ap': float(test_ap),
            'test_f1': float(test_f1),
            'test_f2': float(test_f2),
            'test_recall': float(test_rec),
        },
        'training_config': {
            'epochs': config.EPOCHS,
            'lr': config.LR,
            'focal_alpha': config.FOCAL_ALPHA,
            'focal_gamma': config.FOCAL_GAMMA,
            'pos_weight': float(pos_weight),
            'fraud_oversample': config.FRAUD_OVERSAMPLE,
        },
    }

    with open(config.OUT_PKL, 'wb') as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model, artifact, best_threshold


#Run
if __name__ == "__main__":
    model, artifact, threshold = train_gnn_model()
