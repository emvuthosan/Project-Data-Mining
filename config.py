from import_library import *

class Config:
    def __init__(self):
        self.DATA_ROOT = '/kaggle/input/datasets/trunghongquc/dataset/data' #MUST change this data link 
        self.EDGE_DIR = os.path.join(self.DATA_ROOT, 'edge_df_csv')
        self.FEAT_DIR = os.path.join(self.DATA_ROOT, 'features_df_csv')
        self.NODE_DIR = os.path.join(self.DATA_ROOT, 'node_df_csv')

        self.OUT_DIR = '/kaggle/working/models'
        self.VERSION = datetime.now().strftime("%Y%m%d_%H%M")
        self.OUT_PKL = f'{self.OUT_DIR}/gnn_model.pkl'

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SEED = 42

        self.HIDDEN = 128      
        self.HEADS = 4
        self.DROPOUT = 0.50     

        self.EPOCHS = 80          
        self.LR = 3e-4        
        self.WD = 1e-5       
        self.PATIENCE = 15          

        self.FOCAL_ALPHA = 0.99     
        self.FOCAL_GAMMA = 3.0     

        self.USE_CLASS_WEIGHT = True
        self.FRAUD_OVERSAMPLE = 30  

        self.NUM_NEIGHBORS = [10, 5]  
        self.TRAIN_BATCH_SIZE = 1024      
        self.INF_BATCH_SIZE = 4096
        self.NUM_WORKERS = 2

        self.MIN_RECALL = 0.70
        self.BETA = 2.0


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')