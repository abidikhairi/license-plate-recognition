import random
import matplotlib.pyplot as plt
from torchvision import transforms as T
from datasets import LicensePlateDetectionDataset


def show_image_with_bbox(img, bbox):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor='red', linewidth=3))
    plt.show()


if __name__ == '__main__':
    root_dir = 'data/license_plates_detection_train'
    metadata_file = 'data/license_plates_detection_train.csv'
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    dataset = LicensePlateDetectionDataset(root_dir, metadata_file, transform)
    idx = random.randint(0, len(dataset))
    img, bbox = dataset[idx]
    
    show_image_with_bbox(img, bbox)
    
    # target_size = 224
    # _height = img.shape[1]
    # _width = img.shape[2]
    
    # x_scale = target_size / _width
    # y_scale = target_size / _height
    
    # xmax = int(bbox[0] * x_scale)
    # ymax = int(bbox[1] * y_scale)
    # xmin = int(bbox[2] * x_scale)
    # ymin = int(bbox[3] * y_scale)
    
    # print("scale: ", x_scale, y_scale)
    # print("original bbox:" , bbox[0], bbox[1], bbox[2], bbox[3])
    # print("scaled bbox:", xmin, ymin, xmax, ymax)

    # transform = T.Compose([
    #     T.Resize((224, 224)),
    #     T.ToTensor(),
    # ])

    # dataset = LicensePlateDetectionDataset(root_dir, metadata_file, transform)
    # img, bbox = dataset[idx]
    # show_image_with_bbox(img, [xmax, ymax, xmin, ymin])
