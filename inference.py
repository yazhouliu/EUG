import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import os
import time
import cv2 as cv
from exp import getExp
from exp_config import cfg
import argparse
# warnings.filterwarnings('ignore', '.*imshow.*', )

def file_filter(f):
    if f[-4:] in ['.jpg']:
        return True
    else:
        return False
    
def file_finder(in_dir):
    
    files = os.listdir(in_dir)
    
    files = list(filter(file_filter,files))
    return files

def output_colormap(img, path_grey, path_color):

    #img = (img-img.min())/(img.max()-img.min())
    
    im_gray = (img*255).astype('uint8')
    im_color =  cv.applyColorMap(im_gray, cv.COLORMAP_JET)
    cv.imwrite(path_color,im_color)
    cv.imwrite(path_grey,im_gray)

def load_imgs(src_dir,device):

    image_mean = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std) 
    image_std = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std) 
    transform1 = Compose([ToTensor(), Normalize(image_mean, image_std)])
    
    img_list=[]    
    img_files=file_finder(src_dir)
    
    for idx, file in enumerate(img_files):
        print('loading image: {}/{}'.format(idx+1,len(img_files)))
        img_path = os.path.join(src_dir,file)
        
        image = Image.open(img_path)#.convert('RGB')

        #..for lf data, image should be smaller to fit to the memory
        image = image.resize((1024,512))


        image = transform1(image).unsqueeze(0).to(device)
        
        img_list.append(image)

    return img_list, img_files

def det_anomaly(model, img_list):

    start_time = time.time()

    model.eval()
    anomaly_score_list=[]

    with torch.no_grad():
        for idx,img in enumerate(img_list):
            print('detecting anomaly: {}/{}'.format(idx+1,len(img_list)))
            out = model(img)
            anomaly_score = out['anomaly_score'][0,0]
            anomaly_score = anomaly_score.cpu().numpy()
            anomaly_score_list.append(anomaly_score)

    print("total seconds:%d"%(time.time() - start_time))

    return anomaly_score_list


def main(args):
    

    img_dir=args.img_dir
    out_dir=args.out_dir
    # img_dir = '/media/yazhou/data_drive1/wxy3/segData/ra'
    # out_dir = '/home/yazhou/wxy4/EUG/output.ra'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fuse_decoder_ckpt_path=args.ckpt_path
    # fuse_decoder_ckpt_path='/home/yazhou/wxy4/remake/psp_ocr_laf_train/psp+ocr_laf_train/checkpoints/checkpoint-best.pth'
    
    device=torch.device('cuda:0')
    model = getExp(cfg)
    model.load_checkpoint(fuse_decoder_ckpt_path)
    model=model.to(device)
    model.eval()

    img_list, img_files = load_imgs(img_dir,device)
    
    res_list=det_anomaly(model, img_list)
    
    #..save the results
    for idx, file in enumerate(img_files):
        gname = file.replace('.jpg','grey.jpg')
        cname = file.replace('.jpg','colr.jpg')

        path_grey =  os.path.join(out_dir,gname)
        path_color=  os.path.join(out_dir,cname)

        output_colormap(res_list[idx], path_grey, path_color)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Inference Images')
    parser.add_argument('ckpt_path', type=str, default=None)
    parser.add_argument('out_dir', type=str, default=None)
    parser.add_argument('img_dir', type=str, default=None)
    args=parser.parse_args()
    main(args)