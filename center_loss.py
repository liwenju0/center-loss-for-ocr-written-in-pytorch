import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    IGNORE_INDEX = 0
    def __init__(self, num_classes, feat_dim, use_gpu=True, alpha=0.05, verbose=False, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.verbose = verbose
        self.device = device
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(self.device), requires_grad=False)
        self.centers = torch.load("Your_path/center_loss_centers.pth")
        self.centers = self.centers.to(self.device)
        self.centers.requires_grad = False

    def forward(self, predicts, batch):
        if not isinstance(predicts, (list, tuple)   ):
            raise ValueError(f"predicts must be a list or tuple, but got {type(predicts)}")
        predicts, embedding = predicts
        predicts.to(self.device)
        embedding.to(self.device)

        labels, label_len = batch[1], batch[2]
        labels.to(self.device)
        label_len.to(self.device)

        
        
        
        # 准备特征和标签
        features, labels, poses = self.prepare_feature_labels(embedding, predicts, labels, label_len)
        
        # 计算中心损失
        loss = self.compute_center_loss(features, labels)
        ret_dict = {"loss": loss}
        
        # 更新中心
        self.update_centers(features, labels)
        
        # 如果启用详细模式，添加额外信息
        if self.verbose:
            verbose_dict = self.center_verbose(features, labels, poses, embedding, predicts, batch)
            ret_dict.update(verbose_dict)
        
        return ret_dict
    
    
    def center_verbose(self, features, labels, poses, embedding, predicts, batch):
        """
        打印各种距离变化的过程。加上 verbose 有利于观察变化过程，但可能训练变慢、占用空间变大
        """
        with torch.no_grad():
            labels_brother, label_prob = [], []
            raw_prob = predicts.softmax(dim=2)
            top2_probs, top2_pred = torch.topk(raw_prob, k=2, dim=2)
            # 找到和真实值不同的、预测概率最大的字符，即所谓形近字
            for i in range(poses.shape[0]):
                pos = poses[i]
                label = labels[i]
                near_pred = top2_pred[pos[0], pos[1], 0]
                gt_prob = raw_prob[pos[0], pos[1], label]
                if near_pred.item() == label.item():
                    near_pred = top2_pred[pos[0], pos[1], 1]
                labels_brother.append(near_pred)
                label_prob.append(gt_prob)
            
            labels_brother = torch.tensor(labels_brother, dtype=torch.int64)
            label_prob = torch.tensor(label_prob, dtype=torch.float32)
            centers_brother = self.centers[labels_brother]
            centers_batch = self.centers[labels]
            # 计算形近字类别中心之间的距离
            center_distance = torch.norm(centers_batch - centers_brother, dim=-1).mean()
            # 计算字符到形近字中心的距离
            to_brother_center = torch.norm(features - centers_brother, dim=-1)
            distance_to_brother = to_brother_center.mean()
            # 计算字符到自己类别中心的距离
            to_self_center = torch.norm(features - centers_batch, dim=-1)
            distance_to_self = to_self_center.mean()
            diff = to_brother_center - to_self_center
            # 计算距离最大与最小的字符，距离指的是 字符距自身中心的距离 - 字符距形近字中心的距离
            values, indices = torch.topk(diff, k=3)
            indices = indices.to(self.device)
            max_k = torch.gather(labels, 0, indices)
            label_prob = label_prob.to(self.device)
            max_probs = torch.gather(label_prob, 0, indices)
            
            values, indices = torch.topk(-diff, k=3)
            indices = indices.to(self.device)
            labels = labels.to(self.device)
            min_k = torch.gather(labels, 0, indices)
            min_probs = torch.gather(label_prob, 0, indices)
            
            return {
                "区分最好的字符": max_k,
                "区分最好的字符概率": max_probs,
                "区分最差的字符": min_k,
                "区分最差的字符概率": min_probs,
                "形近字类别中心之间的距离": center_distance,
                "字符到形近字中心的距离": distance_to_brother,
                "字符到自己类别中心的距离": distance_to_self
            }


        

    def compute_center_loss(self, features, labels):
        assert features.shape[0] == labels.shape[0], f"features.shape[0] != labels.shape[0] ({features.shape[0]} != {labels.shape[0]})"
        labels = labels.reshape(-1)
        centers_batch = self.centers[labels]
        loss = torch.nn.functional.mse_loss(features, centers_batch)
        return loss
    
    def ctc_mask(self, raw_pred):
        """
        根据ctc的预测结果，构造mask，mask掉重复的字符,blank字符，空格字符，返回mask
        """
        with torch.no_grad():
            char_rep = (raw_pred[:, :-1] == raw_pred[:, 1:])
            
            # 将前一个位置设为mask
            char_rep = torch.cat([torch.zeros_like(char_rep[:, :1], dtype=torch.bool), char_rep], dim=1)
            
            # 构造 IGNORE_INDEX 的mask
            is_char = raw_pred > self.IGNORE_INDEX
            
            # 两个mask进行and操作，得到最终的mask
            char_no_rep = torch.logical_and(is_char, ~char_rep)

            return char_no_rep
        

    def prepare_feature_labels(self, embedding, predicts, labels,  label_len):
        """
        rawlogits: shape [batch_size, seq_len, num_classes]
        labels: shape [batch_size, seq_len]
        embedding: shape [batch_size, seq_len, embedding_dim]
        char_num: shape [batch_size]
        得到用于计算 centerloss 的 embedding features，和对应的标签
        忽略掉切出的字符数量与真实值不同的样本
        返回用于计算 centerloss 的 embedding features，和对应的标签
        """
        res_features = []
        res_labels = []
        res_poses = []
        with torch.no_grad():
            raw_pred = torch.argmax(predicts, axis=2)
            char_no_rep = self.ctc_mask(raw_pred)
            batch_size = raw_pred.shape[0]
            for i in range(batch_size):
                char_num_i = label_len[i]
                char_no_rep_i = char_no_rep[i]
                char_no_rep_i = torch.where(char_no_rep_i)[0]
                embedding_i = embedding[i]
                labels_i = labels[i]
                if char_num_i.item() != char_no_rep_i.shape[0]:
                    # print(f"切出的字符数量与真实值不同，忽略此样本：{labels_i[:char_num_i.item()]}, {torch.sum(char_no_rep_i).item()} vs {char_num_i.item()}")
                    continue
                else:
                    res_features.append(embedding_i[char_no_rep_i])
                    res_labels.append(labels_i[:char_num_i.item()])
                    for seq_idx in char_no_rep_i:
                        res_poses.append((i, seq_idx))
        if not res_features:  # 检查res_features是否为空
            # 返回默认值,避免后续操作出错
            return torch.zeros(0, self.feat_dim), torch.zeros(0, dtype=torch.int64), torch.zeros((0, 2), dtype=torch.int64)
        
        res_features = torch.concat(res_features, dim=0)
        res_labels = torch.concat(res_labels, dim=0)
        res_poses = torch.tensor(res_poses, dtype=torch.int64)

        assert res_features.shape[0] == res_poses.shape[0], \
            f"res_features.shape[0] != res_poses.shape[0] ({res_features.shape[0]} != {res_poses.shape[0]})"
        assert res_features.shape[0] == res_labels.shape[0], \
            f"res_features.shape[0] != res_labels.shape[0] ({res_features.shape[0]} != {res_labels.shape[0]})"
        
        return res_features, res_labels, res_poses
    
    def update_centers(self, features, labels):
        """
        更新中心
        features: shape [num_samples, embedding_dim]
        labels: shape [num_samples]
        
        """
        assert features.shape[0] == labels.shape[0], f"features.shape[0] != labels.shape[0] ({features.shape[0]} != {labels.shape[0]})"
        labels = labels.reshape(-1)
       # 计算每个标签的出现次数
        unique_labels, counts = torch.unique(labels, return_counts=True)
        # 对每个唯一的标签进行更新
        for label, count in zip(unique_labels, counts):
            mask = (labels == label)
            label_features = features[mask]
            
            # 计算新的中心
            new_center = (1 - self.alpha) * self.centers[label] + self.alpha * label_features.mean(dim=0)
            with torch.no_grad():   
                # 更新中心
                self.centers[label] = new_center

class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, tuple):
            predicts, features = predicts
        else:
            features = None
        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        # assert torch.all(label < len(alphabet)), "Some labels are out of range"
        predicts += 1e-10
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1 - weight
            weight = torch.square(weight)
            loss = loss * weight
        loss = loss.mean()
        return loss
    

class EnhancedCTCLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.1, use_focal_loss=False, center_verbose=True, **kwargs):
        super(EnhancedCTCLoss, self).__init__()
        self.device = kwargs.get('device', 'cpu')
        # print(f"device: {self.device}")
        self.center_verbose = center_verbose
        self.ctc_loss = CTCLoss(use_focal_loss)
        self.center_loss = CenterLoss(num_classes, feature_dim, alpha, verbose=center_verbose, device=self.device)
        
        self.lambda_center = kwargs.get('lambda_center', 0.1)  # 中心损失的权重

    def forward(self, predicts, batch):
        original_predicts = predicts
        assert isinstance(predicts, tuple), "predicts must be a tuple"
        predicts, features = predicts
        ctc_loss = self.ctc_loss(predicts, batch)
        center_loss = self.center_loss(original_predicts, batch)  # 假设batch[1]是标签
        if isinstance(ctc_loss, dict):
            ctc_loss = ctc_loss['loss']
        total_loss = ctc_loss + self.lambda_center * center_loss['loss']
        
        res_dict = {
            'loss': total_loss,
            'ctc_loss': ctc_loss,
            'center_loss': center_loss['loss']
        }
        center_loss.pop('loss')
        res_dict.update(center_loss)
        return res_dict