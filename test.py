from utils.getter import *
from torch.utils.data import Dataset, DataLoader
import timm
import argparse
import os

model_archs = ['efficientnet_b1', 'vit_base_patch16_224']
weights_path = sorted(os.listdir('weights'))
weights = [1, 1, 1, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDataset(Dataset):
    """
    Reads a folder of images
    """

    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.dir = img_dir
        self.file_names = df['image_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img_path = os.path.join(self.dir, img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        if self.transforms:
            item = self.transforms(image=img)
            img = item['image']

        return img


class EnsembleModel(nn.Module):
    def __init__(
        self,
        num_classes,
        name="vit_base_patch16_224",
        from_pretrained=False
    ):
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained=from_pretrained)
        if name.find("efficientnet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        elif name.find("resnext") != -1:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif name.find("vit") != -1:
            self.model.head = nn.Linear(
                self.model.head.in_features, num_classes)
        elif name.find("densenet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )

    def forward(self, x):
        return self.model(x)


def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        image_preds = model(imgs)
        image_preds_all += [torch.softmax(image_preds,
                                          1).detach().cpu().numpy()]
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def test(config):
    df = pd.DataFrame()
    df['image_id'] = os.listdir(config.test_imgs)
    print('df', df.head())

    for i, model_arch in enumerate(model_archs):
        if model_arch.split('_')[0] == 'vit':
            test_vit_transforms = get_augmentation(config, _type='vit_test')
            testset = TestDataset(
                df=df, img_dir=config.test_imgs, transforms=test_vit_transforms)
        else:
            test_transforms = get_augmentation(config, _type='test')
            testset = TestDataset(
                df=df, img_dir=config.test_imgs, transforms=test_transforms)
        testloader = DataLoader(testset, batch_size=config.batch_size,
                                num_workers=config.num_workers, pin_memory=True, shuffle=False)

        model = EnsembleModel(num_classes=config.num_classes,
                              name=model_arch).to(device)

        tst_preds = []
        for j, weight in enumerate(weights_path[i*2: i*2 + 2]):
            states = torch.load(
                os.path.join('weights', weight))
            model.load_state_dict(states['model'], strict=False)
            with torch.no_grad():
                for _ in range(2):
                    tst_preds += [weights[j] / sum(weights) / 2 *
                                  inference_one_epoch(model, testloader, device)]

        avg_tst_preds = np.mean(tst_preds, axis=0)

        if not (os.path.isdir('./total_preds')):
            os.mkdir('./total_preds')
        np.save('./total_preds/total_preds.npy', tst_preds)

        if not (os.path.isdir('./mean_preds')):
            os.mkdir('./mean_preds')
        np.save('./mean_preds/mean_preds.npy', avg_tst_preds)

        del model
        torch.cuda.empty_cache()

    df['label'] = np.argmax(avg_tst_preds, axis=1)
    df.to_csv('submission.csv', index=False)
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Testing Emsemble (efficientnet-b1 & vit_base_patch16_224)')
    parser.add_argument('--config', default='test', type=str,
                        help='project file that contains parameters')
    args = parser.parse_args()
    config = Config(os.path.join('configs', args.config + '.yaml'))

    test(config)
