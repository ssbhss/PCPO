import os
import os.path as osp
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import silhouette_score

# Global device constant to avoid hard-coded .cuda() calls
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint

import clip
from network.ResNet import ResNet

# Import torch_kmeans for clustering; fall back to sklearn KMeans if missing
try:
    from torch_kmeans import KMeans
except ImportError:
    from sklearn.cluster import KMeans as SKLearnKMeans

    class KMeans:
        """Simple wrapper around sklearn KMeans to keep a compatible API.

        Notes:
        - Accepts torch.Tensor inputs and returns torch tensors on DEVICE.
        - Maps `num_init` -> `n_init` for sklearn.
        """
        def __init__(self, init_method="k-means++", num_init=10, n_clusters=2, verbose=False):
            self.n_clusters = n_clusters
            # sklearn uses n_init
            self.sklearn_kmeans = SKLearnKMeans(n_clusters=n_clusters, init=init_method, n_init=max(1, num_init), verbose=verbose)

        def _to_numpy_2d(self, x):
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = np.asarray(x)
            if x_np.ndim == 1:
                x_np = x_np.reshape(-1, 1)
            elif x_np.ndim > 2:
                x_np = x_np.reshape(-1, x_np.shape[-1])
            return x_np

        def fit(self, x):
            x_np = self._to_numpy_2d(x)
            self.sklearn_kmeans.fit(x_np)
            return self

        def predict(self, x):
            x_np = self._to_numpy_2d(x)
            labels = self.sklearn_kmeans.predict(x_np)
            return torch.tensor(labels, device=DEVICE)


class CLIPPreprocessedDataset(Dataset):
    """Dataset wrapper that applies CLIP preprocessing"""
    def __init__(self, base_dataset, clip_preprocess):
        self.base_dataset = base_dataset
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get original data from DASSL dataset
        data = self.base_dataset[idx]

        if isinstance(data, dict):
            # For DASSL datasets that return dict
            # Get the raw image path and load with CLIP preprocess
            img_path = data["impath"]
            from PIL import Image
            pil_img = Image.open(img_path).convert("RGB")
            # Use the provided CLIP preprocess (already CPU-based transform)
            clip_img = self.clip_preprocess(pil_img)

            # Return with CLIP-processed image
            return {
                "img": clip_img,
                "label": data["label"],
                "impath": data["impath"]
            }
        else:
            # Handle other formats - fallback to original
            return data


class IndexedDataset(Dataset):
    """Dataset wrapper that adds index to each sample"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get original data
        data = self.base_dataset[idx]
        
        # Add index to the data
        if isinstance(data, dict):
            # For DASSL datasets that return dict
            data["idx"] = idx
            return data
        elif isinstance(data, tuple):
            # For tuple-based datasets, add index at the end
            return (*data, idx)
        else:
            # For other cases, create a dict
            return {"data": data, "idx": idx}


def entropy_weighted_average(*outputs):
    """Entropy-weighted average of multiple outputs"""
    def calc_entropy(output):
        return -torch.sum(output * torch.log(output), dim=1)

    entropies = [calc_entropy(output) for output in outputs]
    weights = [torch.exp(-entropy).unsqueeze(1) for entropy in entropies]
    weighted_outputs = [weight * output for weight, output in zip(weights, outputs)]
    return torch.stack(weighted_outputs).sum(dim=0) / torch.stack(weights).sum(dim=0)


def get_initc(features, outputs):
    """Initialize class centroids"""
    features_ = features / (features.norm(2, 1, keepdim=True))
    initc = outputs.t() @ features_
    initc /= 1e-8 + outputs.sum(dim=0)[:, None]
    return initc


def get_confidence_idx(outputs, num_classes):
    """Get confidence indices using K-means clustering on entropy"""
    confidence_idx = []
    for i in range(num_classes):
        idx = torch.where(outputs.argmax(1) == i)[0]
        if len(idx) <= 2:
            continue

        # Calculate normalized entropy for the selected indices
        ent = torch.sum(-outputs[idx] * torch.log(outputs[idx]), dim=1) / math.log(num_classes)
        ent = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent))
        ent = ent.to(DEVICE)

        # Use K-means to cluster entropy values
        num_init = 10 if len(idx) > 10 else max(1, len(idx) - 1)
        ent_2d = ent.view(-1, 1)
        kmeans = KMeans(init_method="k-means++", num_init=num_init, n_clusters=2, verbose=False).fit(ent_2d)
        labels = kmeans.predict(ent_2d).squeeze()

        # Select low-entropy (high-confidence) samples
        idx_1 = torch.where(labels == 1)[0]
        iidx = 0
        if ent[idx_1].mean() > ent.mean():
            iidx = 1
        confidence_idx_ = idx[torch.where(labels != iidx)[0]]
        # Extend with CPU-side ints to avoid device issues
        confidence_idx.extend(confidence_idx_.cpu().tolist())

    if len(confidence_idx) == 0:
        return torch.tensor([], device=DEVICE, dtype=torch.long)

    return torch.tensor(confidence_idx, device=DEVICE, dtype=torch.long)


@TRAINER_REGISTRY.register()
class PCPO(TrainerX):
    """Prototype-based Cross-domain Prompt Optimization for Black-box Domain Adaptation"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PCPO.ARCH in ["ViT-B/32", "RN50"]
        assert cfg.TRAINER.PCPO.SOURCE_ARCH in ["resnet50", "resnet101"]

    # def _get_class_names(self, dataset_name):
    #     """Get class names for the dataset"""
    #     if dataset_name == "Office31":
    #         return ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair',
    #                 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard',
    #                 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug',
    #                 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder',
    #                 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
    #
    #     elif dataset_name == "OfficeHome":
    #         return ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
    #                 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
    #                 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
    #                 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade',
    #                 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
    #                 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio',
    #                 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda',
    #                 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
    #
    #     elif dataset_name == "VisDA17":
    #         return ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
    #                 'skateboard', 'train', 'truck']
    #
    #     elif dataset_name == "miniDomainNet":
    #         # miniDomainNet 126 classes - using a subset of common classes
    #         # In practice, you might want to load these from a file
    #         return ['aircraft_carrier', 'alarm_clock', 'ant', 'anvil', 'asparagus', 'axe', 'banana',
    #                 'basket', 'bathtub', 'bear', 'bee', 'bird', 'blackberry', 'blueberry', 'bottlecap',
    #                 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera',
    #                 'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello',
    #                 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow',
    #                 'crab', 'crocodile', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck',
    #                 'dumbbell', 'elephant', 'eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower',
    #                 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter',
    #                 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster',
    #                 'microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'peanut',
    #                 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple', 'potato',
    #                 'power_outlet', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone',
    #                 'screwdriver', 'sea_turtle', 'see_saw', 'sheep', 'shoe', 'skateboard', 'snake', 'speedboat',
    #                 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'swan',
    #                 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower',
    #                 'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'truck', 'umbrella', 'vase',
    #                 'watermelon', 'whale', 'zebra']
    #
    #     else:
    #         # Generic class names as fallback
    #         return [f"class_{i}" for i in range(self.num_classes)]

    def build_model(self):
        cfg = self.cfg

        # Get dataset-specific configuration
        dataset_name = cfg.DATASET.NAME

        # Normalize dataset name (handle case sensitivity issues)
        dataset_name_mapping = {
            "office31": "Office31",
            "Office31": "Office31",
            "office_home": "OfficeHome",
            "OfficeHome": "OfficeHome",
            "visda17": "VisDA17",
            "VisDA17": "VisDA17",
            "minidomainnet": "miniDomainNet",
            "miniDomainNet": "miniDomainNet"
        }

        # Normalize the dataset name
        normalized_name = dataset_name_mapping.get(dataset_name, dataset_name)
        if normalized_name != dataset_name:
            print(f"Warning: Dataset name normalized from '{dataset_name}' to '{normalized_name}'")
            dataset_name = normalized_name

        # Map DASSL dataset names to our config names
        dataset_mapping = {
            "Office31": "Office31",
            "OfficeHome": "OfficeHome",
            "VisDA17": "VisDA17",
            "miniDomainNet": "miniDomainNet"
        }

        config_name = dataset_mapping.get(dataset_name, dataset_name)

        if config_name in cfg.TRAINER.PCPO.DATASET_CONFIGS:
            dataset_config = cfg.TRAINER.PCPO.DATASET_CONFIGS[config_name]
            self.num_classes = dataset_config.NUM_CLASSES
            self.source_arch = dataset_config.SOURCE_ARCH
        else:
            # Fallback to default values
            self.num_classes = cfg.TRAINER.PCPO.NUM_CLASSES
            self.source_arch = cfg.TRAINER.PCPO.SOURCE_ARCH

        print(f"Building PCPO for {dataset_name}")
        print(f"Classes: {self.num_classes}, Source arch: {self.source_arch}")
        print(f"CLIP model: {cfg.TRAINER.PCPO.ARCH}")
        print(f"Learning rate: {cfg.TRAINER.PCPO.LEARNING_RATE}")
        print(f"Weight decay: {cfg.TRAINER.PCPO.WEIGHT_DECAY}")
        print(f"Momentum: {cfg.TRAINER.PCPO.MOMENTUM}")
        print(f"Prototype epochs: {cfg.TRAINER.PCPO.PROTOTYPE_EPOCHS}")

        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(cfg.TRAINER.PCPO.ARCH, device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Load pre-trained source model (API)
        source_model_path = cfg.TRAINER.PCPO.SOURCE_MODEL_PATH
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"Source model not found: {source_model_path}")

        print(f"Loading source model from: {source_model_path}")

        if self.source_arch == "resnet101":
            self.api_model = ResNet(101, self.num_classes, pretrained=False)
            self.pretrained_model = ResNet(101, self.num_classes, pretrained=True).to(self.device).eval()
        else:
            self.api_model = ResNet(50, self.num_classes, pretrained=False)
            self.pretrained_model = ResNet(50, self.num_classes, pretrained=True).to(self.device).eval()

        # Load source model weights
        checkpoint = torch.load(source_model_path, map_location=self.device, weights_only=True)
        self.api_model.load_state_dict(checkpoint, strict=False)
        self.api_model = self.api_model.to(self.device).eval()

        # Freeze API model
        for p in self.api_model.parameters():
            p.requires_grad = False

        # Setup class categories for CLIP text features
        self.categories = self.dm.dataset.classnames

        # Convert categories to lowercase and replace underscores with spaces (as in source code)
        self.categories = [c.lower().replace('_', ' ') for c in self.categories]

        print("PCPO models loaded successfully")

    def build_data_loader(self):
        """Build data loader using DASSL standard approach"""
        # Use DASSL's standard data loading
        super().build_data_loader()

        # For PCPO, we need target domain data with indices AND CLIP preprocessing
        # Load CLIP model temporarily to get preprocess function
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            original_dataset = self.test_loader.dataset

            # Reuse preprocess from the already loaded CLIP model if available
            clip_preprocess = getattr(self, 'preprocess', None)
            if clip_preprocess is None:
                _, clip_preprocess = clip.load(self.cfg.TRAINER.PCPO.ARCH, device=self.device)

            # Create CLIP-preprocessed dataset
            clip_dataset = CLIPPreprocessedDataset(original_dataset, clip_preprocess)
            indexed_dataset = IndexedDataset(clip_dataset)

            self.target_loader = DataLoader(
                indexed_dataset,
                batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
                shuffle=False,  # No shuffling for inference
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                # Only pin memory when using CUDA for slight performance benefit
                pin_memory=(self.device.type == 'cuda')
            )

            # Store dataset info
            self.num_samples = len(indexed_dataset)
            print(f"Target dataset loaded: {self.num_samples} samples, {self.num_classes} classes")
            print(f"Using CLIP preprocessing for target domain data")
        else:
            raise RuntimeError("No test loader found. PCPO requires target domain data.")

    def obtain_clip_labels(self):
        """Obtain CLIP predictions for target domain"""
        print("Obtaining CLIP predictions...")

        self.clip_model.eval()

        # Get text features for all classes
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c.lower().replace('_', ' ')}.")
            for c in self.categories
        ]).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs).float()

        # Get image features and predictions
        if self.cfg.TRAINER.PCPO.ARCH == 'RN50':
            image_features = torch.empty(self.num_samples, 1024).to(self.device)
        else:
            image_features = torch.empty(self.num_samples, 512).to(self.device)

        all_labels = torch.empty(self.num_samples).to(self.device).long()

        with torch.no_grad():
            for batch in self.target_loader:
                # Handle DASSL data format
                if isinstance(batch, dict):
                    # Images are already CLIP-preprocessed
                    inputs = batch["img"].to(self.device)
                    labels = batch["label"].to(self.device)
                    indices = batch["idx"].to(self.device)
                else:
                    # Handle tuple format (img, label, idx)
                    inputs, labels, indices = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    indices = indices.to(self.device)

                # Extract CLIP features
                feas = self.clip_model.encode_image(inputs).float()
                image_features[indices.long(), :] = feas
                all_labels[indices.long()] = labels

        # Compute similarities and predictions
        image_features_ = image_features / (torch.norm(image_features, 2, 1, keepdim=True))
        text_features_ = text_features / (torch.norm(text_features, 2, 1, keepdim=True))
        clip_outputs = image_features_ @ text_features_.t()
        clip_outputs_softmax = nn.Softmax(dim=1)(clip_outputs * self.clip_model.logit_scale.exp())

        clip_predict_label = torch.max(clip_outputs, 1)[1]
        clip_accuracy = torch.sum(clip_predict_label == all_labels) / len(all_labels) * 100
        print(f"CLIP prediction accuracy: {clip_accuracy:.4f}%")

        return clip_outputs_softmax, image_features, all_labels

    def obtain_api_labels(self):
        """Obtain API (source model) predictions for target domain"""
        print("Obtaining API predictions...")

        self.api_model.eval()

        all_fea = torch.empty(self.num_samples, 2048).to(self.device)
        all_output = torch.empty(self.num_samples, self.num_classes).to(self.device)
        all_label = torch.empty(self.num_samples).to(self.device).long()

        with torch.no_grad():
            for batch in self.target_loader:
                # Handle DASSL data format
                if isinstance(batch, dict):
                    inputs = batch["img"].to(self.device)
                    labels = batch["label"].to(self.device)
                    indices = batch["idx"].to(self.device)
                else:
                    # Handle tuple format (img, label, idx)
                    inputs, labels, indices = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    indices = indices.to(self.device)

                # Get features and outputs from ResNet model
                features = self.pretrained_model.get_features(inputs)
                outputs = self.api_model(inputs)

                all_fea[indices.long(), :] = features
                all_output[indices.long(), :] = outputs
                all_label[indices.long()] = labels

        predict = torch.max(all_output, 1)[1]
        accuracy = torch.sum(predict == all_label).item() / float(all_label.size()[0])
        all_output = nn.Softmax(dim=1)(all_output)
        all_fea /= torch.norm(all_fea, p=2, dim=1, keepdim=True)

        # Adjust the source model predictions using confidence-based centroids
        initc, _ = self.get_confidence_initc(all_fea, all_output)
        # initc = get_initc(all_fea, all_output)
        initc_ = initc / (torch.norm(initc, p=2, dim=1, keepdim=True))
        all_fea_norm = all_fea / (torch.norm(all_fea, p=2, dim=1, keepdim=True))
        initc_outputs = all_fea_norm @ initc_.t()
        initc_predict_label = torch.max(initc_outputs, 1)[1]
        fix_accuracy = torch.sum(initc_predict_label == all_label).item() / float(all_label.size()[0])

        print(f"API accuracy: {accuracy*100:.4f} -> {fix_accuracy*100:.4f}")

        fix_outputs = nn.Softmax(dim=1)((initc_outputs - 1) / 0.07).detach()
        fusion_label_ = entropy_weighted_average(all_output, fix_outputs)
        fusion_accuracy = torch.sum(fusion_label_.argmax(1) == all_label) / len(all_label) * 100
        print(f"Fusion accuracy: {fusion_accuracy:.4f}%")

        return fusion_label_, all_label, all_fea

    def get_confidence_initc(self, features, outputs):
        """Get confidence-based initial centroids"""
        confidence_idx = []
        initc = torch.zeros_like(get_initc(features, outputs))
        predict = torch.max(outputs, 1)[1]

        for i in range(self.num_classes):
            idx = torch.where(predict == i)[0]
            if len(idx) <= 2:
                initc[i, :] = get_initc(features, outputs)[i, :]
                continue

            # Calculate normalized entropy
            ent = torch.sum(-outputs[idx] * torch.log(outputs[idx]), dim=1) / math.log(self.num_classes)
            ent = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent))
            ent = ent.cuda()

            # Use K-means clustering
            num_init = 10 if len(idx) > 10 else len(idx) - 1
            # Reshape entropy to 2D for K-means (N samples, 1 feature)
            ent_2d = ent.view(-1, 1)
            kmeans = KMeans(init_method="k-means++", num_init=num_init, n_clusters=2, verbose=False).fit(ent_2d)
            labels = kmeans.predict(ent_2d).squeeze()

            # Select high-confidence samples
            idx_1 = torch.where(labels == 1)[0]
            iidx = 0
            if ent[idx_1].mean() > ent.mean():
                iidx = 1
            confidence_idx_ = idx[torch.where(labels != iidx)[0]]
            confidence_idx.extend(confidence_idx_)

            # Compute weighted centroid
            if len(confidence_idx_) > 0:
                weights = outputs[confidence_idx_, :].max(dim=1)[0][:, None]
                weighted_features = features[confidence_idx_, :] * weights
                initc[i, :] = (torch.sum(weighted_features, dim=0) /
                              (torch.sum(weights, dim=0).item())).unsqueeze(0)
            else:
                initc[i, :] = get_initc(features, outputs)[i, :]

        return initc, confidence_idx

    def lr_scheduler(self, optimizer, iter_num, max_iter, learning_rate=1, gamma=10, power=0.75):
        """Learning rate scheduler"""
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        cfg = self.cfg
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * decay
            param_group['weight_decay'] = cfg.TRAINER.PCPO.WEIGHT_DECAY
            param_group['momentum'] = cfg.TRAINER.PCPO.MOMENTUM
        return optimizer

    def get_domain_prototype(self, all_features, init_prototype, target_idx, labels, pseudo_labels, ls=0.1):
        """Optimize domain-specific prototypes"""
        batch_size = 65536

        # Place prototype parameters on the trainer device
        fix_text_features = nn.Parameter(init_prototype.clone().detach().to(self.device))
        fix_text_features.requires_grad = True

        cfg = self.cfg
        optimizer_fix = torch.optim.SGD([fix_text_features],
                                       lr=cfg.TRAINER.PCPO.LEARNING_RATE,
                                       weight_decay=cfg.TRAINER.PCPO.WEIGHT_DECAY,
                                       momentum=cfg.TRAINER.PCPO.MOMENTUM)

        # Normalize and detach features (do not backprop through features)
        all_features = all_features / (torch.norm(all_features, 2, 1, keepdim=True))
        all_features = all_features.detach()

        epochs = cfg.TRAINER.PCPO.PROTOTYPE_EPOCHS
        max_iter = epochs * (len(target_idx) // batch_size + 1)
        iter_num = 0

        print(f"Prototype optimization: {epochs} epochs, {len(target_idx)} samples, batch_size={batch_size}")
        print(f"Max iterations: {max_iter}")

        for epoch in range(epochs):
            for i in range((len(target_idx) // batch_size) + 1):
                iter_num += 1
                optimizer_fix = self.lr_scheduler(optimizer_fix, iter_num, max_iter,
                                                  cfg.TRAINER.PCPO.LEARNING_RATE, 10, 0.75)
                optimizer_fix.zero_grad()

                fix_text_features_ = fix_text_features / (torch.norm(fix_text_features, 2, 1, keepdim=True))

                # Safe slicing of target indices
                batch_indices = target_idx[i * batch_size: (i + 1) * batch_size]
                if batch_indices.numel() == 0:
                    continue

                logits = all_features[batch_indices.long()] @ fix_text_features_.t()
                loss = nn.CrossEntropyLoss(label_smoothing=ls)(
                    logits * 100,
                    pseudo_labels[batch_indices.long()].argmax(1)
                )

                loss.backward()
                optimizer_fix.step()

        fix_text_features_ = fix_text_features / (torch.norm(fix_text_features, 2, 1, keepdim=True))
        logits = all_features @ fix_text_features_.t()
        clip_outputs = nn.Softmax(dim=1)(logits * 100)

        return fix_text_features.detach(), clip_outputs.detach()

    def refine_predictions_high_to_middle(self, pseudo_label, all_label, all_fea, high_idx, middle_idx):
        """Refine predictions from high confidence to middle confidence samples"""
        all_idx = torch.arange(len(middle_idx), device=self.device)
        known_idx = torch.where(torch.isin(middle_idx, high_idx))[0]
        unknown_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]

        fea = all_fea / (torch.norm(all_fea, 2, 1, keepdim=True))

        while len(unknown_idx) > 0:
            if len(unknown_idx) > 10:
                confirm_num = len(unknown_idx) // 2
            else:
                confirm_num = len(unknown_idx)

            # Compute initc from current pseudo labels and update EMA teacher
            initc = self.compute_and_update_initc(fea, pseudo_label)
            _, predict = self.get_domain_prototype(all_fea[middle_idx], initc, known_idx,
                                                 all_label[middle_idx], pseudo_label[middle_idx])
            initc = get_initc(fea, pseudo_label)
            predict_idx = predict_max_value.sort(descending=True)[1]
            confirm_idx = predict_idx[:confirm_num]
            new_known_idx = unknown_idx[confirm_idx]

            pseudo_label[middle_idx[new_known_idx]] = predict[confirm_idx]
            known_idx = torch.cat((known_idx, new_known_idx))
            unknown_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]

        # Calculate accuracy
        accuracy = torch.sum(pseudo_label[middle_idx].argmax(1) == all_label[middle_idx]).item() / len(middle_idx)
        print(f"High->Middle refinement accuracy: {accuracy*100:.4f}%")

    def refine_predictions_high_middle_to_low(self, pseudo_label_1, fea_1, pseudo_label_2, fea_2,
                                            all_label, high_middle_idx):
        """Refine predictions from high+middle confidence to low confidence samples"""
        all_idx = torch.arange(len(all_label), device=self.device)
        known_idx = high_middle_idx
        unknown_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]

        fea_1_ = fea_1 / (torch.norm(fea_1, 2, 1, keepdim=True))
        fea_2_ = fea_2 / (torch.norm(fea_2, 2, 1, keepdim=True))

        while len(unknown_idx) > 0:
            # Compute and update EMA prototypes from both views; teacher EMA is shared/updated
            initc = self.compute_and_update_initc(fea_1_, pseudo_label_1)
            initc_ = self.compute_and_update_initc(fea_2_, pseudo_label_2)

            _, predict = self.get_domain_prototype(fea_1_, initc, known_idx, all_label, pseudo_label_1)
            _, predict_ = self.get_domain_prototype(fea_2_, initc_, known_idx, all_label, pseudo_label_1)

            predict = predict[unknown_idx]
            predict_ = predict_[unknown_idx]

            # Find samples where both models agree
            confirm_idx = torch.where(predict.argmax(1) == predict_.argmax(1))[0]

            if len(confirm_idx) != 0:
                new_known_idx = unknown_idx[confirm_idx]
                new_label = entropy_weighted_average(predict[confirm_idx], predict_[confirm_idx])
                pseudo_label_1[new_known_idx] = new_label
                known_idx = torch.cat((known_idx, new_known_idx))
                unknown_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]
            else:
                # If no agreement, use entropy-weighted average for all remaining
                new_label = entropy_weighted_average(predict, predict_)
                pseudo_label_1[unknown_idx] = new_label
                break

        # Calculate final accuracy
        accuracy = torch.sum(pseudo_label_1.argmax(1) == all_label).item() / len(all_label)
        print(f"High+Middle->Low refinement accuracy: {accuracy*100:.4f}%")

    def train(self):
        """Main PCPO training/inference process"""
        print("Starting PCPO inference...")
        
        # Step 1: Get CLIP predictions
        clip_label, image_features = self.get_clip_logits()
        
        # Step 2: Obtain API predictions
        api_label, all_label, all_fea = self.obtain_api_labels()
        
        # Step 3: Identify confidence levels
        equal_idx = torch.where(api_label.argmax(1) == clip_label.argmax(1))[0]
        api_confidence_idx = get_confidence_idx(api_label, self.num_classes)
        clip_confidence_idx = get_confidence_idx(clip_label, self.num_classes)
        
        equal_idx_api = torch.unique(torch.cat((equal_idx, api_confidence_idx)))
        equal_idx_clip = torch.unique(torch.cat((equal_idx, clip_confidence_idx)))
        
        print("=" * 60)
        print("Refining High confidence samples to Middle confidence samples")
        print("=" * 60)
        
        # Step 4: Refine high -> middle confidence
        print("Refining API predictions...")
        self.refine_predictions_high_to_middle(api_label, all_label, all_fea, equal_idx, equal_idx_api)
        
        print("Refining CLIP predictions...")
        self.refine_predictions_high_to_middle(clip_label, all_label, image_features, equal_idx, equal_idx_clip)
        
        print("=" * 60)
        print("Refining High and Middle confidence samples to Low confidence samples")
        print("=" * 60)
        
        # Step 5: Refine high+middle -> low confidence
        print("Refining API with CLIP guidance...")
        self.refine_predictions_high_middle_to_low(api_label, all_fea, clip_label, image_features, 
                                                 all_label, equal_idx_api)
        
        print("Refining CLIP with API guidance...")
        self.refine_predictions_high_middle_to_low(clip_label, image_features, api_label, all_fea, 
                                                 all_label, equal_idx_clip)
        
        # Step 6: Final inference using silhouette analysis
        api_acc = torch.sum(torch.max(api_label, 1)[1] == all_label).item() / len(all_label)
        clip_acc = torch.sum(torch.max(clip_label, 1)[1] == all_label).item() / len(all_label)
        
        print("=" * 60)
        print("Final Inference")
        print("=" * 60)
        
        # Calculate silhouette scores
        silhouette_api = silhouette_score(all_fea.cpu().numpy(), api_label.argmax(1).cpu().numpy())
        silhouette_clip = silhouette_score(image_features.cpu().numpy(), clip_label.argmax(1).cpu().numpy())
        silhouette_api_clip = silhouette_score(all_fea.cpu().numpy(), clip_label.argmax(1).cpu().numpy())
        silhouette_clip_api = silhouette_score(image_features.cpu().numpy(), api_label.argmax(1).cpu().numpy())
        
        # Choose the best prediction based on silhouette scores
        if silhouette_api + silhouette_clip_api > silhouette_clip + silhouette_api_clip:
            final_label = api_label
            final_acc = api_acc
            final_description = "API"
        else:
            final_label = clip_label
            final_acc = clip_acc
            final_description = "CLIP"
        
        print(f"Final Results: API: {api_acc * 100:.4f}%, CLIP: {clip_acc * 100:.4f}%")
        print(f"Selected: {final_description} with accuracy: {final_acc * 100:.4f}%")
        
        # Save results
        self.save_results(api_acc, clip_acc, final_acc, final_description)
        
        return final_acc

    def save_results(self, api_acc, clip_acc, final_acc, final_description):
        """Save experiment results"""
        results_file = osp.join(self.output_dir, "final_results.txt")
        
        with open(results_file, "w") as f:
            f.write(f"PCPO Final Results\n")
            f.write(f"==================\n")
            f.write(f"Domain: {self.cfg.DATASET.NAME}\n")
            f.write(f"Source: {self.cfg.DATASET.SOURCE_DOMAINS[0]}\n")
            f.write(f"Target: {self.cfg.DATASET.TARGET_DOMAINS[0]}\n")
            f.write(f"API Accuracy: {api_acc * 100:.4f}%\n")
            f.write(f"CLIP Accuracy: {clip_acc * 100:.4f}%\n")
            f.write(f"Final Method: {final_description}\n")
            f.write(f"Final Accuracy: {final_acc * 100:.4f}%\n")
        
        print(f"Results saved to: {results_file}")

    def test(self, split=None):
        """Test method for compatibility with DASSL"""
        # PCPO doesn't have a separate test phase, return the final accuracy
        if hasattr(self, 'final_accuracy'):
            return self.final_accuracy * 100
        else:
            return 0.0