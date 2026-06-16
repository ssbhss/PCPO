import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import json
import math
import random
import torch
from torchvision import  datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from torch_kmeans import KMeans
from datetime import datetime
import shutil
import clip
from sklearn.metrics import silhouette_score
from network.ResNet import ResNet
from PIL import Image
from classifier import Classifier

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class IndexedDataset(Dataset):
    def __init__(self, data_dir, transform=None, target='test'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = target == 'train'

        # 读取image_list.txt文件
        with open(os.path.join(self.data_dir, "image_list.txt"), "r") as f:
            self.files = [line.rstrip() for line in f if line != ""]

        self.samples = []
        for item in self.files:
            img_path, label = item.split(' ')
            self.samples.append((os.path.join(data_dir, img_path), int(label)))

        if self.shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        # 读取图片
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return (img, label), index

    def __len__(self):
        return len(self.samples)

def getClipLoader(preprocess, args):
    domain = './data/' + args.domain
    data_dir = os.path.join(domain, args.target_domain)
    data = IndexedDataset(data_dir, transform=preprocess)
    data_loader = DataLoader(data, batch_size=2048, shuffle=False, num_workers=0)
    return data_loader


class ExternalInputIteratorIndices(object):
    def __init__(self, batch_size, data_dir, tar):
        self.images_dir = data_dir
        self.batch_size = batch_size
        with open(os.path.join(self.images_dir, "image_list.txt"), "r") as f:
            self.files = [os.path.join(data_dir, line.rstrip()) for line in f if line != ""]
        self.shuffle = tar == 'train'
        self.indices = list(range(len(self.files)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        self.num_samples = len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.files):
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch, labels, indices = [], [], []
        for i in batch_indices:
            img_path, label = self.files[i].split(' ')
            f = open(os.path.join(img_path), 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
            indices.append(np.array([i], dtype=np.int32))
            # TODO: [lable]和[i]改成label和i
            self.current_index += 1
        return (batch, labels, indices)


if HAS_DALI:
    class ImagePipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, external_data, device):
            super(ImagePipeline, self).__init__(batch_size, num_threads, device_id)
            self.external_data = [(x, y, z) for x, y, z in external_data]
            self.shuffle = external_data.shuffle
            self.num_samples = external_data.num_samples
            self.device = device

        def define_graph(self):
            self.jpegs, self.labels, self.indices = fn.external_source(source=self.external_data, num_outputs=3,
                                                                       device="cpu", cycle='raise')
            if self.device == 'gpu':
                images = fn.decoders.image(self.jpegs, device="mixed", output_type=types.RGB)
            else:
                images = fn.decoders.image(self.jpegs, device="cpu", output_type=types.RGB)
            if self.shuffle:
                images = fn.resize(images, antialias=True, resize_x=256, resize_y=256, device=self.device)
                # images = fn.resize(images, antialias=True, resize_x=256, resize_y=256, interp_type=DALIInterpType.INTERP_CUBIC, device=self.device)
                outputs = fn.crop_mirror_normalize(images, dtype=types.DALIDataType.FLOAT, mirror=1, crop=(224, 224),
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255], device=self.device)
            else:
                images = fn.resize(images, resize_x=224, resize_y=224, antialias=True, device=self.device)
                # images = fn.resize(images, resize_x=224, resize_y=224, interp_type=DALIInterpType.INTERP_CUBIC, antialias=True, device=self.device)
                outputs = fn.crop_mirror_normalize(images, dtype=types.DALIDataType.FLOAT, crop=(224, 224),
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255], output_layout='CHW',
                                                   device=self.device)
            return outputs, self.labels, self.indices
else:
    class ImagePipeline(object):
        pass


def load_data(root_path, domain, batch_size, tar, device='gpu'):
    if HAS_DALI:
        iterator = ExternalInputIteratorIndices(batch_size, os.path.join(root_path, domain), tar)
        pipeline = ImagePipeline(batch_size, 12, 0, iterator, device)
        pipeline.build()
        return DALIGenericIterator(pipeline, ['data', 'labels', 'indices'], auto_reset=True, last_batch_padded=True,
                                   last_batch_policy=LastBatchPolicy.PARTIAL)
    else:
        from torchvision import transforms
        
        class PyTorchDALIWrapper:
            def __init__(self, dataloader):
                self.dataloader = dataloader
            def __iter__(self):
                for (imgs, labels), indices in self.dataloader:
                    yield [{"data": imgs, "labels": labels, "indices": indices}]
            def __len__(self):
                return len(self.dataloader)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_dir = os.path.join(root_path, domain)
        dataset = IndexedDataset(data_dir, transform=transform, target=tar)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(tar == 'train'), num_workers=0)
        return PyTorchDALIWrapper(dataloader)


def get_loader(args):
    data_dir = os.path.join('./data', args.domain)
    test_loader = load_data(data_dir, args.target_domain, args.batch_size, 'test')
    return test_loader


def getAPI(args):
    if args.domain == 'visda17':
        api = ResNet(101, args.num_classes, pretrained=False)
    else:
        api = ResNet(50, args.num_classes, pretrained=False)
    api.fc = torch.nn.Linear(api.fc.in_features, args.num_classes)
    api.load_state_dict(torch.load('./best_model/best_resnet_' + args.source_domain + '.pth', map_location='cuda:0', weights_only=True), strict=False)
    return api.cuda().eval()


def Entropy(input_):
    epsilon = 1e-8
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_initc(features, outputs, args):
    features_ = features / (features.norm(2, 1, keepdim=True) + 1e-8)
    initc = outputs.t() @ features_
    initc /= 1e-8 + outputs.sum(dim=0)[:, None]
    return initc


def get_simi(a, b):
    a_ = a / (a.norm(2, 1, keepdim=True) + 1e-8)
    b_ = b / (b.norm(2, 1, keepdim=True) + 1e-8)
    return a_ @ b_.t()


def get_confidence_initc(features, outputs, args):
    all_initc = get_initc(features, outputs, args)
    confidence_idx = []
    initc = torch.zeros_like(all_initc)
    predict = torch.max(outputs, 1)[1]
    for i in range(args.num_classes):
        idx = torch.where(predict == i)[0]
        if len(idx) <= 2:
            initc[i, :] = all_initc[i, :]
            continue
        ent = torch.sum(-outputs[idx] * torch.log(outputs[idx] + 1e-8), dim=1) / math.log(args.num_classes)
        ent = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent) + 1e-8)
        ent = ent.cuda()
        num_init = 10 if len(idx) > 10 else len(idx) - 1
        kmeans = KMeans(init_method="k-means++", num_init=num_init, n_clusters=2, verbose=False).fit(
            ent.view(-1, 1).unsqueeze(0))
        labels = kmeans.predict(ent.reshape(-1, 1).unsqueeze(0)).squeeze()
        idx_1 = torch.where(labels == 1)[0]
        iidx = 0
        if ent[idx_1].mean() > ent.mean():
            iidx = 1
        confidence_idx_ = idx[torch.where(labels != iidx)[0]]
        confidence_idx.extend(confidence_idx_)
        initc[i, :] = (torch.sum(features[confidence_idx_, :] * outputs[confidence_idx_, :].max(dim=1)[0][:, None],
                                 dim=0) / (
                               torch.sum(outputs[confidence_idx_, :].max(dim=1)[0][:, None], dim=0).item() + 1e-8)).unsqueeze(0)
    return initc, confidence_idx


def get_confidence_idx(outputs, args):
    confidence_idx = []
    for i in range(args.num_classes):
        idx = torch.where(outputs.argmax(1) == i)[0]
        if len(idx) <= 2:
            continue
        ent = torch.sum(-outputs[idx] * torch.log(outputs[idx] + 1e-8), dim=1) / math.log(args.num_classes)
        ent = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent) + 1e-8)
        ent = ent.cuda()
        num_init = 10 if len(idx) > 10 else len(idx) - 1
        kmeans = KMeans(init_method="k-means++", num_init=num_init, n_clusters=2, verbose=False).fit(
            ent.view(-1, 1).unsqueeze(0))
        labels = kmeans.predict(ent.reshape(-1, 1).unsqueeze(0)).squeeze()
        idx_1 = torch.where(labels == 1)[0]
        iidx = 0
        if ent[idx_1].mean() > ent.mean():
            iidx = 1
        confidence_idx_ = idx[torch.where(labels != iidx)[0]]
        confidence_idx.extend(confidence_idx_)
    return torch.tensor(confidence_idx).cuda()


def obtain_api_labels(api, model, loader, args):
    api.eval()
    model.eval()
    all_fea = torch.empty(args.num_samples, 2048).cuda()
    all_output = torch.empty(args.num_samples, args.num_classes).cuda()
    all_label = torch.empty(args.num_samples).cuda().long()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]['data'].cuda()
            labels = batch[0]['labels'].cuda().long().view(-1)
            j = batch[0]['indices'].view(-1)
            all_fea[j.long(), :] = model(inputs, True)[0]
            all_output[j.long(), :] = api(inputs)
            all_label[j.long()] = labels
            # print(labels)
            # print(torch.max(all_output[j.long(), :], 1)[1])

    predict = torch.max(all_output, 1)[1].cuda()
    accuracy = torch.sum(predict == all_label).item() / float(all_label.size()[0])
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea /= torch.norm(all_fea, p=2, dim=1, keepdim=True) + 1e-8

    # Adjust the source model predictions
    initc, _ = get_confidence_initc(all_fea, all_output, args)
    initc_ = initc / (torch.norm(initc, p=2, dim=1, keepdim=True) + 1e-8)
    all_fea /= torch.norm(all_fea, p=2, dim=1, keepdim=True) + 1e-8
    initc_outputs = all_fea @ initc_.t()
    initc_predict_label = torch.max(initc_outputs, 1)[1].cuda()
    fix_accuracy = torch.sum(initc_predict_label == all_label).item() / float(all_label.size()[0])
    print(f"Accuracy: {accuracy*100:.4f} -> {fix_accuracy*100:.4f}")
    fix_outputs = nn.Softmax(dim=1)((initc_outputs - 1) / 0.07).detach()
    fusion_label_ = entropy_weighted_average(all_output, fix_outputs)
    print(f"fusion accuracy: {torch.sum(fusion_label_.argmax(1) == all_label) / len(all_label) * 100}")
    return fusion_label_, all_label, all_fea


def obtain_label_by_clip(loader, model, args):
    model.eval()
    image_features = torch.empty(args.num_samples, 512).cuda()
    if args.arch == ('RN50'):
        image_features = torch.empty(args.num_samples, 1024).cuda()
    all_labels = torch.empty(args.num_samples).cuda().long()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0][0].cuda()
            labels = batch[0][1].cuda()
            idx = batch[1].cuda()
            feas = model.encode_image(inputs).float()
            image_features[idx.long(), :] = feas
            all_labels[idx.long()] = labels

        text_features = obtain_clip_text_features(model, args)

    image_features_ = image_features / (torch.norm(image_features, 2, 1, keepdim=True) + 1e-8)
    text_features_ = text_features / (torch.norm(text_features, 2, 1, keepdim=True) + 1e-8)
    clip_outputs = image_features_ @ text_features_.t()
    clip_outputs_softmax = nn.Softmax(dim=1)(clip_outputs * 100)
    clip_predict_label = torch.max(clip_outputs, 1)[1].cuda()
    print(f"clip_predict: {torch.sum(clip_predict_label == all_labels) / len(all_labels) * 100}")
    return clip_outputs_softmax, image_features, text_features


def obtain_clip_text_features(model, args):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in args.categories]).cuda()
    return model.encode_text(text_inputs).float().cuda()


def lr_scheduler_(optimizer, iter, max_iter, args, learning_rate=1, gamma=10, power=0.75):
    decay = (1 + gamma * iter / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * decay
        param_group['weight_decay'] = args.weight_decay
        param_group['momentum'] = args.momentum
    return optimizer


def topn_accuracy(outputs, target, topk=(1,)):
    if len(target.size()) > 1:
        target = target.argmax(1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct_k.mul_(100.0 / batch_size)
        res.append(correct_k)
        print(f"Top-{k} Accuracy: {correct_k.item()}")
    return res


def entropy_weighted_average(*outputs):
    def calc_entropy(output):
        return -torch.sum(output * torch.log(output + 1e-8), dim=1)

    entropies = [calc_entropy(output) for output in outputs]
    weights = [torch.exp(-entropy).unsqueeze(1) for entropy in entropies]
    weighted_outputs = [weight * output for weight, output in zip(weights, outputs)]
    return torch.stack(weighted_outputs).sum(dim=0) / torch.stack(weights).sum(dim=0)


def get_domain_prototype(all_features, init_prototype, target_idx, labels, pesu_labels, args, ls=0.1, verbose=False):
    batch_size = 65536
    fix_text_features = nn.Parameter(init_prototype.clone().detach()).cuda()
    fix_text_features.requires_grad = True
    optimizer_fix = torch.optim.SGD([fix_text_features], lr=args.learning_rate, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    all_features /= torch.norm(all_features, 2, 1, keepdim=True) + 1e-8
    all_features = all_features.detach()
    epochs = 200
    max_iter = epochs * (len(target_idx) // batch_size + 1)
    iter_num = 0
    for _ in range(epochs):
        total_acc = 0
        total_loss = 0
        for i in range((len(target_idx) // batch_size) + 1):
            iter_num += 1
            optimizer_fix = lr_scheduler_(optimizer_fix, iter_num, max_iter, args, args.learning_rate, 10, 0.75)
            optimizer_fix.zero_grad()
            fix_text_features_ = fix_text_features / (torch.norm(fix_text_features, 2, 1, keepdim=True) + 1e-8)
            try:
                logits = all_features[target_idx[i * batch_size: (i + 1) * batch_size]] @ fix_text_features_.t()
                loss = nn.CrossEntropyLoss(label_smoothing=ls)(logits * 100,
                                                               pesu_labels[target_idx[
                                                                           i * batch_size: (i + 1) * batch_size]].argmax(
                                                                   1))
                total_acc += torch.sum(
                    torch.max(logits, 1)[1] == pesu_labels[target_idx[i * batch_size: (i + 1) * batch_size]].argmax(
                        1)).item()
            except:
                logits = all_features[target_idx[i * batch_size:]] @ fix_text_features_.t()
                loss = nn.CrossEntropyLoss(label_smoothing=ls)(logits * 100,
                                                               pesu_labels[target_idx[i * batch_size:]].argmax(1))
                total_acc += torch.sum(
                    torch.max(logits, 1)[1] == pesu_labels[target_idx[i * batch_size:]].argmax(1)).item()
            loss.backward()
            optimizer_fix.step()
            total_loss += loss.item()
    fix_text_features_ = fix_text_features / (torch.norm(fix_text_features, 2, 1, keepdim=True) + 1e-8)
    logits = all_features @ fix_text_features_.t()
    if verbose:
        print(f"True Acc: {torch.sum(torch.max(logits, 1)[1] == labels).item() / len(labels)}")
    clip_outputs = nn.Softmax(dim=1)(logits * 100)
    return fix_text_features.detach(), clip_outputs.detach()


def refine_predictions_from_high2middel(args, pseudo_label, all_label, all_fea, high_id, middel_idx, verbose=False):
    all_idx = torch.arange(len(middel_idx)).cuda()
    known_idx = torch.where(torch.isin(middel_idx, high_id))[0]
    un_known_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]
    fea = all_fea / (torch.norm(all_fea, 2, 1, keepdim=True) + 1e-8)
    while len(un_known_idx) > 0:
        if len(un_known_idx) > 10:
            confirm_num = len(un_known_idx) // 2
        else:
            confirm_num = len(un_known_idx)
        initc = get_initc(fea, pseudo_label, args)
        _, predict = get_domain_prototype(all_fea[middel_idx], initc, known_idx, all_label[middel_idx],
                                          pseudo_label[middel_idx], args)
        predict = predict[un_known_idx]
        predict_max_value, predict_max_idx = torch.max(predict, 1)
        predict_idx = predict_max_value.sort(descending=True)[1]
        confirm_idx = predict_idx[:confirm_num]
        new_known_idx = un_known_idx[confirm_idx]
        if verbose:
            print(len(new_known_idx))
            topn_accuracy(pseudo_label[middel_idx][new_known_idx], all_label[middel_idx][new_known_idx])
            topn_accuracy(predict[confirm_idx], all_label[middel_idx][new_known_idx])
        pseudo_label[middel_idx[new_known_idx]] = predict[confirm_idx]
        known_idx = torch.cat((known_idx, new_known_idx))
        un_known_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]
        if verbose:
            topn_accuracy(pseudo_label[middel_idx], all_label[middel_idx])
    topn_accuracy(pseudo_label[middel_idx], all_label[middel_idx])


def refine_predictions_from_high_middel2low(args, pseudo_label_1, fea_1, pseudo_label_2, fea_2, all_label, high_middel_idx, verbose=False):
    all_idx = torch.arange(len(all_label)).cuda()
    known_idx = high_middel_idx
    un_known_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]
    fea_1_ = fea_1 / (torch.norm(fea_1, 2, 1, keepdim=True) + 1e-8)
    fea_2_ = fea_2 / (torch.norm(fea_2, 2, 1, keepdim=True) + 1e-8)
    while len(un_known_idx) > 0:
        initc = get_initc(fea_1_, pseudo_label_1, args)
        initc_ = get_initc(fea_2_, pseudo_label_2, args)
        _, predict = get_domain_prototype(fea_1_, initc, known_idx, all_label, pseudo_label_1, args)
        _, predict_ = get_domain_prototype(fea_2_, initc_, known_idx, all_label, pseudo_label_1, args)
        predict = predict[un_known_idx]
        predict_ = predict_[un_known_idx]
        confirm_idx = torch.where(predict.argmax(1) == predict_.argmax(1))[0]
        if len(confirm_idx) != 0:
            new_known_idx = un_known_idx[confirm_idx]
            new_label = entropy_weighted_average(predict[confirm_idx], predict_[confirm_idx])
            if verbose:
                print(len(new_known_idx))
                topn_accuracy(pseudo_label_1[new_known_idx], all_label[new_known_idx])
                topn_accuracy(new_label, all_label[new_known_idx])
            pseudo_label_1[new_known_idx] = new_label
            known_idx = torch.cat((known_idx, new_known_idx))
            un_known_idx = all_idx[torch.isin(all_idx, known_idx, invert=True)]
            if verbose:
                topn_accuracy(pseudo_label_1, all_label)
        else:
            new_label = entropy_weighted_average(predict, predict_)
            if verbose:
                topn_accuracy(pseudo_label_1[un_known_idx], all_label[un_known_idx])
                topn_accuracy(new_label, all_label[un_known_idx])
            pseudo_label_1[un_known_idx] = new_label
            if verbose:
                topn_accuracy(pseudo_label_1, all_label)
            break
    topn_accuracy(pseudo_label_1, all_label)


def main():
    seed = 2020
    seed_torch(seed)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    args = argparse.ArgumentParser(description='Model configuration')
    args.add_argument('--config', type=str, default='A-C', help='Configuration file')
    args.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    args.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    args.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    args.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    args.add_argument('--arch', type=str, help='CLIP model architecture')
    args.add_argument('--num_classes', type=int, help='Number of classes')
    args.add_argument('--domain', type=str, help='Domain')
    args.add_argument('--source_domain', type=str, help='Source domain')
    args.add_argument('--target_domain', type=str, help='Target domain')
    args.add_argument('--save_data', type=bool, default=False, help='Save data')
    args.add_argument('--log_path', type=str, default='/log/', help='Log path')
    args.add_argument("--arch_source", type=str, default="resnet50")
    args.add_argument("--bottleneck_dim", type=int, default=256)
    args.add_argument("--weight_norm_dim", type=int, default=0)

    args = args.parse_args()
    path = './config/office31/'
    args.config = 'A-D'
    # path = './config/officehome/'
    # args.config = 'C-A'
    # path = './config/visda17/'
    # args.config = 'visda17'
    # path = './config/domainnet/'
    # args.config = 'P-C'
    cfg = path + args.config + '.json'
    print(cfg)
    with open(cfg, 'r') as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    args.categories = []
    with open(path + 'category.txt', 'r') as f:
        for line in f:
            args.categories.append(line.strip())
    # 将args，categories中全部转化为小写， 并将_替换为空格
    args.categories = [c.lower().replace('_', ' ') for c in args.categories]

    args.arch_source = 'resnet50'
    args.bottleneck_dim = 256
    args.weight_norm_dim = 0
    args.save_data = True
    # args.arch = ('ViT-B/16')
    # args.arch = ('RN50')
    if args.save_data:
        results_dir = args.log_path + args.domain + '/' + args.source_domain + "-" + args.target_domain + "/" + TIMESTAMP
        try:
            os.makedirs(results_dir, exist_ok=True)
        except:
            print("Directory already exists")
        shutil.copyfile(__file__, results_dir + "code.py")
        shutil.copyfile(cfg, results_dir + "config.json")
        sys.stdout = Logger(results_dir + 'log.txt')
    if args.config == 'A-D':
        with open('./results.txt', 'a') as f:
            f.write(f"--{seed}\n")

    clip_model, preprocess = clip.load(args.arch, device="cuda")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_loader = getClipLoader(preprocess, args)
    clip_label, image_features, text_features = obtain_label_by_clip(clip_loader, clip_model, args, )

    if args.domain == 'visda17':
        model_api = ResNet(101, args.num_classes, pretrained=True).cuda()
    else:
        model_api = ResNet(50, args.num_classes, pretrained=True).cuda()
    # api = Classifier(args=args, checkpoint_path='./best_model/best_resnet_' + args.source_domain + '.pth').cuda()
    api = getAPI(args)
    test_loader = get_loader(args)
    api_label, all_label, all_fea = obtain_api_labels(api, model_api, test_loader, args)


    if args.domain == 'visda17':
        api_acc = 0
        clip_acc = 0
        api_acc_ = torch.empty(args.num_classes)
        clip_acc_ = torch.empty(args.num_classes)
        for j in args.categories:
            index = args.categories.index(j)
            api_acc_[index] = torch.sum(torch.max(clip_label[all_label == index], 1)[1] == all_label[all_label == index]).item() / torch.sum(all_label == index)
            print(f"{j} api_acc: {api_acc_[index] * 100}")
            api_acc += api_acc_[index]
        api_acc /= args.num_classes
        print(f"api_acc: {api_acc * 100}")

    # Divide samples according to confidence level
    equal_idx = torch.where(api_label.argmax(1) == clip_label.argmax(1))[0]
    api_confidence_idx = get_confidence_idx(api_label, args)
    clip_confidence_idx = get_confidence_idx(clip_label, args)
    equal_idx_api = torch.unique(torch.cat((equal_idx, api_confidence_idx)))
    equal_idx_clip = torch.unique(torch.cat((equal_idx, clip_confidence_idx)))

    # print("============================================================")
    # print("Refine High confidence samples to Middel confidence samples")
    # print("============================================================")
    # print("Refine API")
    refine_predictions_from_high2middel(args, api_label, all_label, all_fea, equal_idx, equal_idx_api)
    # print("Refine CLIP")
    refine_predictions_from_high2middel(args, clip_label, all_label, image_features, equal_idx, equal_idx_clip)

    # Use High confidence and Middel confidence samples to refine Low confidence samples
    # print("====================================================================")
    # print("Refine High and Middel confidence samples to Low confidence samples")
    # print("====================================================================")
    # print("Refine API")
    refine_predictions_from_high_middel2low(args, api_label, all_fea, clip_label, image_features, all_label, equal_idx_api)
    # print("Refine CLIP")
    refine_predictions_from_high_middel2low(args, clip_label, image_features, api_label, all_fea, all_label, equal_idx_clip)

    # api_acc = torch.sum(torch.max(api_label, 1)[1] == all_label).item() / len(all_label)
    # clip_acc = torch.sum(torch.max(clip_label, 1)[1] == all_label).item() / len(all_label)

    # print("============================================================")
    # print("Inference")
    # print("============================================================")
    # silhouette_api = silhouette_score(all_fea.cpu().numpy(), api_label.argmax(1).cpu().numpy())
    # silhouette_clip = silhouette_score(image_features.cpu().numpy(), clip_label.argmax(1).cpu().numpy())
    # silhouette_api_clip = silhouette_score(all_fea.cpu().numpy(), clip_label.argmax(1).cpu().numpy())
    # silhouette_clip_api = silhouette_score(image_features.cpu().numpy(), api_label.argmax(1).cpu().numpy())
    #
    # if silhouette_api + silhouette_clip_api > silhouette_clip + silhouette_api_clip:
    #     final_label = api_label
    #     final_acc = api_acc
    #     final_description = "API"
    # else:
    #     final_label = clip_label
    #     final_acc = clip_acc
    #     final_description = "CLIP"
    # print(f"Inference :{args.domain} {args.config} API : {api_acc * 100:.4f}, CLIP : {clip_acc * 100:.4f}, Final chose {final_description}: {final_acc}")
    # if args.save_data:
    #     save_path = os.path.join(results_dir, 'results.txt')
    #     with open(save_path, 'a') as f:
    #         f.write(f"{args.domain} {args.config} API : {api_acc * 100:.4f}, CLIP : {clip_acc * 100:.4f}, Final chose {final_description}: {final_acc * 100:.4f}\n")

if __name__ == '__main__':
    main()