from import_library import *
from model_architecture import *

"""
Usage:
    detector = FraudDetector('gnn_model.pkl')

    #Predict on new data
    results = detector.predict(
        node_df   = node_df,      # DataFrame with account_id, bank_id, entity_id
        feat_df   = feat_df,      # DataFrame with account_id + feature columns
        edge_df   = edge_df,      # DataFrame with src, dst, amount, payment_format
        target_ids = [123, 456]   # Optional: only score these account_ids
    )
"""

#Fraud detector
class FraudDetector:
    def __init__(self, pkl_path: str, device: str = 'auto'):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            self.artifact = pickle.load(f)

        #Restore preprocessing objects
        self.scaler = self.artifact['feature_scaler']
        self.le_bank = self.artifact['le_bank']
        self.le_entity = self.artifact['le_entity']
        self.le_pf = self.artifact['le_payment_format']
        self.feat_cols = self.artifact['feat_cols']
        self.n_base = self.artifact['n_base_feats']
        self.cont_col_indices = self.artifact['cont_col_indices']
        self.account_to_idx = self.artifact['account_to_idx']
        self.num_neighbors = self.artifact['num_neighbors']
        self.threshold = self.artifact['threshold']
        self.threshold_method= self.artifact['threshold_method']
        self.model_cfg = self.artifact['model_config']

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #Rebuild & load model
        self.model = HybridGNN(
            in_dim = self.model_cfg['in_dim'],
            hidden = self.model_cfg['hidden'],
            heads = self.model_cfg['heads'],
            dropout = self.model_cfg['dropout'],
        )
        self.model.load_state_dict(self.artifact['model_state_dict'])
        self.model.to(self.device).eval()


    def _engineer_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        df = feat_df.copy()
        df['sent_received_ratio'] = df['total_sent'] / (df['total_received'] + 1e-9)
        df['degree_ratio'] = df['out_degree'] / (df['in_degree'] + 1e-9)
        df['avg_sent'] = df['total_sent'] / (df['out_degree'] + 1e-9)
        df['avg_received'] = df['total_received'] / (df['in_degree'] + 1e-9)
        df['unique_ratio'] = df['unique_senders'] / (df['unique_receivers'] + 1e-9)

        for col in ['total_sent', 'total_received', 'avg_sent', 'avg_received']:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

        return df

    def _safe_encode(self, le, values: pd.Series) -> np.ndarray:
        known = set(le.classes_)
        mapped = values.astype(str).map(
            lambda v: v if v in known else le.classes_[0]
        )
        return le.transform(mapped)

    def _compute_fraud_neighbor_features(self, edge_index_np: np.ndarray, y: np.ndarray, N: int):
        src, dst = edge_index_np[0], edge_index_np[1]
        is_fraud = (y == 1).astype(np.float32)

        fnc = np.zeros(N, dtype=np.float32)
        tnc = np.zeros(N, dtype=np.float32)
        np.add.at(fnc, dst, is_fraud[src])
        np.add.at(tnc, dst, 1.0)

        fnr = fnc / (tnc + 1e-9)
        return fnc, fnr

    def _build_graph(self,
                     node_df: pd.DataFrame,
                     feat_df: pd.DataFrame,
                     edge_df: pd.DataFrame,
                     known_labels: dict = None):
        """
        Args:
            node_df : Must have columns [account_id, bank_id, entity_id]
                            Optionally [label] for evaluation mode
            feat_df : Must have [account_id] + self.feat_cols (base names)
            edge_df : Must have [src, dst, amount, payment_format]
            known_labels : Optional dict {account_id: label} for eval mode

        Returns:
            data : torch_geometric.data.Data
            account_ids : np.ndarray of account_ids (aligned with node index)
            new_to_graph : dict mapping account_id → graph node index
        """

        #Build unified node index
        all_acc = pd.unique(pd.concat([
            node_df['account_id'],
            feat_df['account_id'],
            edge_df['src'],
            edge_df['dst'],
        ], ignore_index=True))

        #Prefer training index, assign new indices for unseen nodes
        new_to_graph = {}
        next_idx = 0
        for acc in all_acc:
            if acc in self.account_to_idx:
                new_to_graph[acc] = self.account_to_idx[acc]
            else:
                #Unseen node: give it a local sequential inde (offset from training N to avoid collision)
                new_to_graph[acc] = acc  #Just a placeholder

        #Remap to compact local indices 0..M-1
        unique_accs = list(all_acc)
        local_idx = {acc: i for i, acc in enumerate(unique_accs)}
        N = len(local_idx)

        #Feature matrix
        n_extra = 4   # bank_id, entity_id, fraud_nb_count, fraud_nb_ratio
        X = np.zeros((N, self.n_base + n_extra), dtype=np.float32)

        feat_df_eng = self._engineer_features(feat_df)
        feat_df_eng['_idx'] = feat_df_eng['account_id'].map(local_idx)
        valid_feat = feat_df_eng.dropna(subset=['_idx'])

        #Fill base features (use only columns that exist)
        available_cols = [c for c in self.feat_cols if c in feat_df_eng.columns]
        missing_cols = [c for c in self.feat_cols if c not in feat_df_eng.columns]
        if missing_cols:
            print(f"Missing feature columns (will be turned into 0 value): {missing_cols}")

        fidx = valid_feat['_idx'].values.astype(int)
        X[fidx, :len(available_cols)] = (
            valid_feat[available_cols].fillna(0).values.astype(np.float32)
        )

        #Categorical features
        node_df_c = node_df.copy()
        node_df_c['_idx'] = node_df_c['account_id'].map(local_idx)
        valid_node = node_df_c.dropna(subset=['_idx'])
        nidx = valid_node['_idx'].values.astype(int)

        X[nidx, self.n_base] = self._safe_encode(self.le_bank,
                                                      valid_node['bank_id'])
        X[nidx, self.n_base + 1] = self._safe_encode(self.le_entity,
                                                      valid_node['entity_id'])

        #Labels (unknown / inference = -1)
        y = np.full(N, -1, dtype=np.int64)
        if 'label' in node_df.columns:
            y[nidx] = valid_node['label'].fillna(-1).values.astype(np.int64)
        if known_labels:
            for acc, lbl in known_labels.items():
                if acc in local_idx:
                    y[local_idx[acc]] = int(lbl)

        #Edges
        edf = edge_df.copy()
        edf['_src'] = edf['src'].map(local_idx)
        edf['_dst'] = edf['dst'].map(local_idx)
        edf = edf.dropna(subset=['_src', '_dst']).copy()
        edf['_src'] = edf['_src'].astype(int)
        edf['_dst'] = edf['_dst'].astype(int)

        edge_agg = (edf.groupby(['_src', '_dst'], sort=False)
                       .agg(total_amount=('amount', 'sum'),
                            tx_count=('amount', 'count'))
                       .reset_index())

        ei_t = torch.tensor(edge_agg[['_src', '_dst']].values.T,
                                  dtype=torch.long)
        edge_index = torch.cat([ei_t, ei_t.flip(0)], dim=1)

        #Fraud-neighbour features
        fnc, fnr = self._compute_fraud_neighbor_features(
            edge_index.numpy(), y, N
        )
        X[:, self.n_base + 2] = fnc
        X[:, self.n_base + 3] = fnr

        #Scale features
        #cont_col_indices from training: base cols + fraud-nb cols
        cont_idx = self.cont_col_indices  # list of int

        #Guard: only apply to columns that exist in current X
        cont_idx_safe = [c for c in cont_idx if c < X.shape[1]]
        X[:, cont_idx_safe] = self.scaler.transform(X[:, cont_idx_safe])

        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
        )

        account_ids = np.array(unique_accs)
        return data, account_ids, local_idx

    @torch.no_grad()
    def predict(self,
                node_df: pd.DataFrame,
                feat_df: pd.DataFrame,
                edge_df: pd.DataFrame,
                target_ids=None,
                threshold: float = None,
                batch_size: int = 4096,
                num_workers: int = 2,
                return_all_nodes: bool = False) -> pd.DataFrame:
        """
        Args:
            node_df        : DataFrame [account_id, bank_id, entity_id, (label)]
            feat_df        : DataFrame [account_id, in_degree, out_degree, ...]
            edge_df        : DataFrame [src, dst, amount, payment_format]
            target_ids     : List/array of account_ids to score.
                             If None, scores all accounts in node_df.
            threshold      : Override the saved threshold (optional)
            batch_size     : Inference batch size
            num_workers    : DataLoader workers
            return_all_nodes: If True, return scores for ALL graph nodes

        Returns:
            pd.DataFrame with columns: account_id | fraud_prob | fraud_pred | risk_tier
        """
        thresh = threshold if threshold is not None else self.threshold

        data, account_ids, local_idx = self._build_graph(
            node_df, feat_df, edge_df
        )
        N = data.x.size(0)

        #Determine which nodes to score
        if return_all_nodes:
            seed_ids = np.arange(N, dtype=np.int64)
            score_acc = account_ids
        else:
            if target_ids is not None:
                score_acc = np.array(target_ids)
            else:
                score_acc = node_df['account_id'].values

            seed_ids = np.array(
                [local_idx[a] for a in score_acc if a in local_idx],
                dtype=np.int64
            )
            score_acc = np.array(
                [a for a in score_acc if a in local_idx]
            )

        if len(seed_ids) == 0:
            print("No target accounts found in graph")
            return pd.DataFrame(columns=['account_id', 'fraud_prob',
                                         'fraud_pred', 'risk_tier'])

        loader = NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            input_nodes=torch.from_numpy(seed_ids),
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        all_probs = []
        self.model.eval()

        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            bs = batch.batch_size
            out = self.model(batch.x, batch.edge_index)[:bs]
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.append(prob)

        fraud_probs = np.concatenate(all_probs)
        fraud_preds = (fraud_probs >= thresh).astype(int)

        results = pd.DataFrame({
            'account_id' : score_acc,
            'fraud_prob' : fraud_probs,
            'fraud_pred' : fraud_preds,
            'risk_tier'  : pd.cut(
                fraud_probs,
                bins  = [0.0,  0.3,   0.5,   0.7,   1.001],
                labels= ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                right = False
            ).astype(str),
        })

        results = results.sort_values('fraud_prob', ascending=False).reset_index(drop=True)
        return results

    @torch.no_grad()
    def predict_single(self,
                       account_id,
                       node_df: pd.DataFrame,
                       feat_df: pd.DataFrame,
                       edge_df: pd.DataFrame,
                       threshold: float = None) -> dict:
        """
        Args:
            account_id : The account to evaluate
            node_df    : Full node context DataFrame
            feat_df    : Full feature DataFrame
            edge_df    : Full edge DataFrame
            threshold  : Override threshold (optional)

        Returns:
            dict with account_id, fraud_prob, fraud_pred, risk_tier, threshold_used, model_version
        """
        results = self.predict(
            node_df = node_df,
            feat_df = feat_df,
            edge_df = edge_df,
            target_ids = [account_id],
            threshold = threshold,
            num_workers = 0,
        )

        if results.empty:
            return {
                'account_id' : account_id,
                'fraud_prob' : None,
                'fraud_pred' : None,
                'risk_tier' : 'UNKNOWN',
                'error' : 'Account not found in graph',
            }

        row = results.iloc[0].to_dict()
        row['threshold_used'] = threshold or self.threshold
        row['model_version']  = self.model_cfg.get('version', 'N/A')
        return row

    @torch.no_grad()
    def evaluate(self,
                 node_df: pd.DataFrame,
                 feat_df: pd.DataFrame,
                 edge_df: pd.DataFrame,
                 threshold: float = None,
                 batch_size: int = 4096,
                 num_workers: int = 2) -> dict:
        """
        Evaluate model on labeled data and return full metrics
        Requires node_df to have a 'label' column

        Returns: dict with auc, ap, f1, f2, recall, precision, confusion_matrix, classification_report, threshold_used
        """

        if 'label' not in node_df.columns:
            raise ValueError("node_df must contain a 'label' column for evaluation")

        thresh = threshold if threshold is not None else self.threshold

        results = self.predict(
            node_df = node_df,
            feat_df = feat_df,
            edge_df = edge_df,
            target_ids = node_df['account_id'].values,
            threshold = thresh,
            batch_size = batch_size,
            num_workers = num_workers,
        )

        #Align labels
        label_map = node_df.set_index('account_id')['label'].to_dict()
        results['true_label'] = results['account_id'].map(label_map)
        results = results.dropna(subset=['true_label'])
        results['true_label'] = results['true_label'].astype(int)

        y_true = results['true_label'].values
        y_prob = results['fraud_prob'].values
        y_pred = results['fraud_pred'].values

        from sklearn.metrics import (roc_auc_score, average_precision_score,
                                     f1_score, fbeta_score, recall_score,
                                     precision_score, classification_report,
                                     confusion_matrix)

        metrics = {
            'auc'                   : float(roc_auc_score(y_true, y_prob)),
            'average_precision'     : float(average_precision_score(y_true, y_prob)),
            'f1'                    : float(f1_score(y_true, y_pred, zero_division=0)),
            'f2'                    : float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
            'recall'                : float(recall_score(y_true, y_pred, zero_division=0)),
            'precision'             : float(precision_score(y_true, y_pred, zero_division=0)),
            'threshold_used'        : thresh,
            'n_samples'             : int(len(y_true)),
            'n_fraud'               : int(y_true.sum()),
            'n_predicted_fraud'     : int(y_pred.sum()),
            'confusion_matrix'      : confusion_matrix(y_true, y_pred).tolist(),
            'classification_report' : classification_report(y_true, y_pred, digits=4),
        }

        print("\nEvaluation Results:")
        print(f"  AUC       : {metrics['auc']:.4f}")
        print(f"  AP        : {metrics['average_precision']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  F2        : {metrics['f2']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"\n{metrics['classification_report']}")
        print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

        return metrics

    def info(self):
        #Print a summary of the loaded artifact
        print("=" * 55)
        print("  FraudDetector – Model Info")
        print("=" * 55)
        print(f"  Version          : {self.model_cfg.get('version','N/A')}")
        print(f"  Input dim        : {self.model_cfg['in_dim']}")
        print(f"  Hidden dim       : {self.model_cfg['hidden']}")
        print(f"  Attention heads  : {self.model_cfg['heads']}")
        print(f"  Dropout          : {self.model_cfg['dropout']}")
        print(f"  Threshold        : {self.threshold:.4f} ({self.threshold_method})")
        print(f"  Min recall target: {self.artifact.get('min_recall_target','N/A')}")
        print(f"  Num neighbors    : {self.num_neighbors}")
        print(f"  Device           : {self.device}")
        if 'training_metrics' in self.artifact:
            m = self.artifact['training_metrics']
            print(f"\n  Training Metrics (test set):")
            print(f"    AUC     : {m.get('test_auc', 'N/A')}")
            print(f"    AP      : {m.get('test_ap',  'N/A')}")
            print(f"    F1      : {m.get('test_f1',  'N/A')}")
            print(f"    Recall  : {m.get('test_recall', 'N/A')}")



#EXAMPLE USAGE
if __name__ == "__main__":
    PKL_PATH = 'gnn_model.pkl'
    DATA_ROOT = '/kaggle/input/datasets/trunghongquc/dataset/data'

    #Load detector
    detector = FraudDetector(PKL_PATH, device='auto')
    detector.info()

    #Load data
    def load_folder(folder):
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    edge_df = load_folder(os.path.join(DATA_ROOT, 'edge_df_csv'))
    feat_df = load_folder(os.path.join(DATA_ROOT, 'features_df_csv'))
    node_df = load_folder(os.path.join(DATA_ROOT, 'node_df_csv'))

    feat_df = feat_df.drop_duplicates(subset=['account_id'], keep='last').reset_index(drop=True)
    node_df = node_df.drop_duplicates(subset=['account_id'], keep='last').reset_index(drop=True)

    #Ex.1: Score ALL accounts in node_df
    results_all = detector.predict(
        node_df = node_df,
        feat_df = feat_df,
        edge_df = edge_df,
    )
    print(results_all.head(10))

    #Ex.2: Score a specific list of account IDs
    #Pick a few account IDs (some fraud, some normal)
    fraud_accounts  = node_df[node_df['label'] == 1]['account_id'].head(5).tolist()
    normal_accounts = node_df[node_df['label'] == 0]['account_id'].head(5).tolist()
    target_accounts = fraud_accounts + normal_accounts

    results_subset = detector.predict(
        node_df    = node_df,
        feat_df    = feat_df,
        edge_df    = edge_df,
        target_ids = target_accounts,
    )
    print(results_subset)

    #Ex.3: Score a single account
    single_id = fraud_accounts[0]
    result_single = detector.predict_single(
        account_id = single_id,
        node_df    = node_df,
        feat_df    = feat_df,
        edge_df    = edge_df,
    )
    print(f"Account: {result_single['account_id']}")
    print(f"Fraud prob: {result_single['fraud_prob']:.4f}")
    print(f"Predicted: {'FRAUD' if result_single['fraud_pred'] else 'NORMAL'}")
    print(f"Risk tier: {result_single['risk_tier']}")

    #Evaluate on labeled test data
    #Use a small labeled subset for quick evaluation
    eval_node_df = node_df.sample(n=min(50_000, len(node_df)), random_state=42)

    metrics = detector.evaluate(
        node_df = eval_node_df,
        feat_df = feat_df,
        edge_df = edge_df,
    )

    print(f"AUC: {metrics['auc']:.4f}")
    print(f"AP: {metrics['average_precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


    #Ex.5: Custom threshold override (incase the optimal threshold not actually OPTIMAL)
    NewThreshold = 0.15
    results_aggressive = detector.predict(
        node_df   = node_df,
        feat_df   = feat_df,
        edge_df   = edge_df,
        threshold = NewThreshold,   #Lower threshold → more fraud flagged → higher recall
    )

    print(f"Flagged with threshold={NewThreshold} : {results_aggressive['fraud_pred'].sum():,}")
    print(f"Flagged with threshold={detector.threshold:.2f} (default): {results_all['fraud_pred'].sum():,}")