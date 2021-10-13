from torchvision import transforms as T

def Aug(train) :
    
    def inner(img):
        
        if train:            
            return T.Compose([T.ToPILImage(),
                              T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
#                              T.Resize((224, 224)), # previous
                              T.RandomHorizontalFlip(p=0.5),
                              T.RandomRotation(degrees=15),
                              T.CenterCrop(size=224),
#                              T.RandomRotation(degrees=45),
                              T.ColorJitter(),
#                              T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img) # Imagenet standard
                    
        else:
            return T.Compose([T.ToPILImage(),
                              T.Resize((224, 224)),
                              T.CenterCrop(size=224),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img) # Imagenet standard dropped
        
    return inner