from net.model import get_psp_ocrnet,get_ocrnet_segformer
import torch
import torch.nn as nn
import torch.nn.functional as F

# ocrnet + segformer 
class Exp_heter(nn.Module):
    def __init__(self,ocr_config_file,ocr_checkpoint_file,segformer_config_file,segformer_checkpoint_file):
        super(Exp_heter,self).__init__()
        self.seg_model_ensemble=get_ocrnet_segformer(ocr_config_file,ocr_checkpoint_file,segformer_config_file,segformer_checkpoint_file)
        self.fuse_decoder=nn.Sequential(nn.Conv2d(256+512+19+19,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                    nn.Conv2d(256,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
                                    nn.Conv2d(16,2,kernel_size=3,padding=1),nn.BatchNorm2d(2),nn.ReLU())
    
    def forward(self,x):
        out=self.seg_model_ensemble(x)
        ocr_feat=out['ocr_feat']
        ocr_seg=out['ocr_seg']
        seg_feat=out['seg_feat']
        seg_seg=out['seg_seg']

        fused_feat=torch.cat((ocr_feat,seg_feat,ocr_seg,seg_seg),dim=1)
        pred=self.fuse_decoder(fused_feat)
        perpixel = F.softmax(pred, dim=1)[:, 0:1, ...]

        pred=F.interpolate(pred,scale_factor=4,mode='bilinear',align_corners=True)
        perpixel=F.interpolate(perpixel,scale_factor=4,mode='bilinear',align_corners=True)

        return {'image':x, 'anomaly_score':perpixel, 'binary_segmentation':pred}
    
    def load_checkpoint(self,path):
        checkpoint=torch.load(path,map_location='cpu')
        self.fuse_decoder.load_state_dict(checkpoint['state_dict'])

# psp + ocr_net 
class Exp_base(nn.Module):
    def __init__(self,psp_config_file,psp_checkpoint_file,ocr_config_file,ocr_checkpoint_file):
        super(Exp_base,self).__init__()
        self.seg_model_ensemble=get_psp_ocrnet(psp_config_file,psp_checkpoint_file,ocr_config_file,ocr_checkpoint_file)
        self.fuse_decoder=nn.Sequential(nn.Conv2d(512+512+19+19,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                    nn.Conv2d(256,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
                                    nn.Conv2d(16,2,kernel_size=3,padding=1),nn.BatchNorm2d(2),nn.ReLU())
    
    def forward(self,x):
        out=self.seg_model_ensemble(x)
        psp_feat=out['psp_feat']
        psp_seg=out['psp_seg']
        ocr_feat=out['ocr_feat']
        ocr_seg=out['ocr_seg']
        
        psp_feat=F.interpolate(psp_feat,scale_factor=2,mode='bilinear',align_corners=True)
        psp_seg=F.interpolate(psp_seg,scale_factor=2,mode='bilinear',align_corners=True)
        fused_feat=torch.cat((psp_feat,ocr_feat,psp_seg,ocr_seg),dim=1)
        pred=self.fuse_decoder(fused_feat)
        perpixel = F.softmax(pred, dim=1)[:, 0:1, ...]

        pred=F.interpolate(pred,scale_factor=4,mode='bilinear',align_corners=True)
        perpixel=F.interpolate(perpixel,scale_factor=4,mode='bilinear',align_corners=True)

        return {'image':x, 'anomaly_score':perpixel, 'binary_segmentation':pred}
    
    def load_checkpoint(self,path):
        checkpoint=torch.load(path,map_location='cpu')
        self.fuse_decoder.load_state_dict(checkpoint['state_dict'])

#psp_small + ocr_small
class Exp_tiny(nn.Module):
    def __init__(self,psp_small_config_file,psp_small_checkpoint_file,ocr_small_config_file,ocr_small_checkpoint_file):
        super(Exp_tiny,self).__init__()
        self.seg_model_ensemble=get_psp_ocrnet(psp_small_config_file,psp_small_checkpoint_file,ocr_small_config_file,ocr_small_checkpoint_file)
        self.fuse_decoder=nn.Sequential(nn.Conv2d(128+512+19+19,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                    nn.Conv2d(256,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
                                    nn.Conv2d(16,2,kernel_size=3,padding=1),nn.BatchNorm2d(2),nn.ReLU())
    
    def forward(self,x):
        out=self.seg_model_ensemble(x)
        psp_feat=out['psp_feat']
        psp_seg=out['psp_seg']
        ocr_feat=out['ocr_feat']
        ocr_seg=out['ocr_seg']
        
        psp_feat=F.interpolate(psp_feat,scale_factor=2,mode='bilinear',align_corners=True)
        psp_seg=F.interpolate(psp_seg,scale_factor=2,mode='bilinear',align_corners=True)
        fused_feat=torch.cat((psp_feat,ocr_feat,psp_seg,ocr_seg),dim=1)
        pred=self.fuse_decoder(fused_feat)
        perpixel = F.softmax(pred, dim=1)[:, 0:1, ...]

        pred=F.interpolate(pred,scale_factor=4,mode='bilinear',align_corners=True)
        perpixel=F.interpolate(perpixel,scale_factor=4,mode='bilinear',align_corners=True)

        return {'image':x, 'anomaly_score':perpixel, 'binary_segmentation':pred}

    def load_checkpoint(self,path):
        checkpoint=torch.load(path,map_location='cpu')
        self.fuse_decoder.load_state_dict(checkpoint['state_dict'])

def getExp(cfg):
    config_file1=cfg.EXPERIMENT.CONFIG_FILE1
    checkpoint_file=cfg.EXPERIMENT.CHECKPOINT_FILE1
    config_file2=cfg.EXPERIMENT.CONFIG_FILE2
    checkpoint_file2=cfg.EXPERIMENT.CHECKPOINT_FILE2
    model_name=cfg.EXPERIMENT.MODEL_NAME
    print(model_name)
    if(model_name == 'eug_tiny'):
        return Exp_tiny(config_file1,checkpoint_file,config_file2,checkpoint_file2)
    elif(model_name == 'eug_base'):
        return Exp_base(config_file1,checkpoint_file,config_file2,checkpoint_file2)
    elif(model_name == 'eug_heter'):
        return Exp_heter(config_file1,checkpoint_file,config_file2,checkpoint_file2)


